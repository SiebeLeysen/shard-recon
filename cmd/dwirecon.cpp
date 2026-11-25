/* Copyright (c) 2017-2019 Daan Christiaens
 *
 * MRtrix and this add-on module are distributed in the hope
 * that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.
 */

#include <algorithm>
#include <sstream>
#include <memory>

#include "command.h"
#include "image.h"
#include "math/SH.h"
#include "dwi/gradient.h"
#include "metadata/phase_encoding.h"
#include "dwi/shells.h"
#include "dwi/svr/shells.h"
#include "adapter/extract.h"

#include "dwi/svr/qspacebasis.h"
#include "dwi/svr/recon.h"
#include "dwi/svr/nnls.h"
#include "dwi/directions/predefined.h"
#include "dwi/svr/parallel.h"

#define DEFAULT_LMAX 4
#define DEFAULT_SSPW 1.0f
#define DEFAULT_REG 0.001
#define DEFAULT_ZREG 0.001
#define DEFAULT_TOL 1e-4
#define DEFAULT_MAXITER 10


using namespace MR;
using namespace App;


void usage ()
{
  AUTHOR = "Daan Christiaens (daan.christiaens@kcl.ac.uk)";

  SYNOPSIS = "Reconstruct DWI signal from a series of scattered slices with associated motion parameters.";

  DESCRIPTION
  + "";

  ARGUMENTS
  + Argument ("DWI", "the input DWI image.").type_image_in()
  + Argument ("SH", "the output spherical harmonics coefficients image.").type_image_out();


  OPTIONS
  + Option ("motion", "The motion parameters associated with input slices or volumes. "
                      "These are supplied as a matrix of 6 columns encoding the rigid "
                      "transformations w.r.t. scanner space in se(3) Lie algebra." )
    + Argument ("file").type_file_in()

  + Option ("rf", "Basis functions for the radial (multi-shell) domain, provided as matrices in which "
                  "rows correspond with shells and columns with SH harmonic bands.").allow_multiple()
    + Argument ("b").type_file_in()
  
  + Option ("response", "Voxel-wise response functions (e.g. from LoRE-SD), as a 5-D image on the reconstruction grid "
                        "(x, y, z, shell, harmonic order l = 0, 2, 4, ...), shells in ascending b-value order as "
                        "classified by this command. The reconstruction then estimates one spherical function per voxel "
                        "(the fibre ODF up to scale), and the signal of shell s is the response times that function, "
                        "order by order. Cannot be combined with -rf; all volumes are used.")
    + Argument ("image").type_image_in()

  + OptionGroup ("Joint reconstruction of the response (with -response)")

  + Option ("atoms", "LoRE-SD atoms (loreatoms output for the shells in this command's order): estimate the response "
                     "jointly with the ODF from the scattered slices. The response of each voxel in -atommask is a "
                     "non-negative mixture of the atoms, initialised by projecting the -response image onto them; the "
                     "reconstruction alternates the ODF step (conjugate gradients) with an atom-weight step "
                     "(approximate Newton steps: per-voxel NNLS on a local quadratic model, exact gradient through the "
                     "slice forward model, line search). Voxels outside the mask keep the given response.")
    + Argument ("K").type_file_in()

  + Option ("atommask", "voxels whose response is estimated (recon grid; required with -atoms)")
    + Argument ("mask").type_image_in()

  + Option ("alternate", "number of response/ODF alternations (default: 2)")
    + Argument ("n").type_integer(0)

  + Option ("atomsteps", "approximate Newton steps per response update (default: 3)")
    + Argument ("n").type_integer(1)

  + Option ("response_out", "output the (final) voxel-wise response, same layout as -response")
    + Argument ("out").type_image_out()

  + Option ("nonneg", "with -response: constrain the ODF to be non-negative (on 129 directions, as LoRE-SD) in voxels "
                      "of -atommask, by ADMM; every ODF step then solves the constrained problem. Fixes the per-order "
                      "split between response and ODF that the data leave open.")

  + Option ("admm", "ADMM iterations per ODF step with -nonneg (default: 8)")
    + Argument ("n").type_integer(1)

  + Option ("admm_cg", "conjugate gradient iterations per ADMM iteration (default: 4)")
    + Argument ("n").type_integer(1)

  + Option ("lmax", "The maximum harmonic order for the output series. (default = " + str(DEFAULT_LMAX) + ")")
    + Argument ("order").type_integer(0, 30)

  + Option ("weights", "Slice weights, provided as a matrix of dimensions Nslices x Nvols.")
    + Argument ("W").type_file_in()

  + Option ("voxweights", "Voxel weights, provided as an image of same dimensions as dMRI data.")
    + Argument ("W").type_image_in()

  + Option ("ssp", "Slice sensitivity profile, either as text file or as a scalar slice thickness for a "
                   "Gaussian SSP, relative to the voxel size. (default = " + str(DEFAULT_SSPW)  + ")")
    + Argument ("w").type_text()

  + Option ("reg", "Isotropic Laplacian regularization. (default = " + str(DEFAULT_REG) + ")")
    + Argument ("l").type_float()

  + Option ("zreg", "Regularization in the slice direction. (default = " + str(DEFAULT_ZREG) + ")")
    + Argument ("l").type_float()

  + Option ("field", "Static susceptibility field, aligned in recon space.")
    + Argument ("map").type_image_in()
    + Argument ("idx").type_integer()

  + Option ("template", "Template header to determine the reconstruction grid.")
    + Argument ("header").type_image_in()

  + DWI::GradImportOptions()

  + Metadata::PhaseEncoding::ImportOptions

  + DWI::ShellsOption

  + OptionGroup ("Output options")

  + Option ("spred",
            "output source prediction of all scattered slices. (useful for diagnostics)")
    + Argument ("out").type_image_out()

  + Option ("odf", "with -response: output the reconstructed per-voxel spherical function (SH coefficients).")
    + Argument ("out").type_image_out()

  + Option ("padding", "zero-padding output coefficients to given dimension.")
    + Argument ("rank").type_integer(0)

  + Option ("complete", "complete (zero-filled) source prediction.")

  + OptionGroup ("CG Optimization options")

  + Option ("tolerance", "the tolerance on the conjugate gradient solver. (default = " + str(DEFAULT_TOL) + ")")
    + Argument ("t").type_float(0.0, 1.0)

  + Option ("maxiter",
            "the maximum number of iterations of the conjugate gradient solver. (default = " + str(DEFAULT_MAXITER) + ")")
    + Argument ("n").type_integer(1)

  + Option ("init",
            "initial guess of the reconstruction parameters.")
    + Argument ("img").type_image_in();

}


typedef float value_type;



void run ()
{
  // Load input data
  auto dwi = Image<value_type>::open(argument[0]).with_direct_io({1, 2, 3, 4});

  // Read motion parameters
  auto opt = get_options("motion");
  Eigen::MatrixXf motion;
  if (opt.size())
    motion = load_matrix<float>(opt[0][0]);
  else
    motion = Eigen::MatrixXf::Zero(dwi.size(3), 6); 

  // Check dimensions
  if (motion.size() && motion.cols() != 6)
    throw Exception("No. columns in motion parameters must equal 6.");
  if (motion.size() && ((dwi.size(3) * dwi.size(2)) % motion.rows()))
    throw Exception("No. rows in motion parameters does not match image dimensions.");


  // Select shells
  auto grad = DWI::get_DW_scheme (dwi);
  DWI::SVR::SVRShells shells (grad);
  shells.select_shells (false, false, false);

  // Read multi-shell basis
  int lmax = 0;
  vector<Eigen::MatrixXf> rf;
  opt = get_options("rf");
  for (size_t k = 0; k < opt.size(); k++) {
    Eigen::MatrixXf t = load_matrix<float>(opt[k][0]);
    if (t.rows() != shells.count())
      throw Exception("No. shells does not match no. rows in basis function " + opt[k][0] + ".");
    lmax = std::max(2*(int(t.cols())-1), lmax);
    rf.push_back(t);
  }

  // Read slice weights
  Eigen::MatrixXf W = Eigen::MatrixXf::Ones(dwi.size(2), dwi.size(3));
  opt = get_options("weights");
  if (opt.size()) {
    W = load_matrix<float>(opt[0][0]);
    if (W.rows() != dwi.size(2) || W.cols() != dwi.size(3))
      throw Exception("Weights matrix dimensions don't match image dimensions.");
  }

  // Read field map and PE scheme
  opt = get_options("field");
  bool hasfield = opt.size();
  auto fieldmap = Image<value_type>();
  size_t fieldidx = 0;
  Eigen::MatrixXf PE;
  if (hasfield) {
    auto petable = Metadata::PhaseEncoding::get_scheme(dwi);
    PE = petable.leftCols<3>().cast<float>();
    PE.array().colwise() *= petable.col(3).array().cast<float>();
    // -----------------------  // TODO: Eddy uses a reverse LR axis for storing
    PE.col(0) *= -1;            // the PE table, akin to the gradient table.
    // -----------------------  // Fix in the eddy import/export functions in core.
    fieldmap = Image<value_type>::open(opt[0][0]);
    fieldidx = opt[0][1];
  }

  // Voxel-wise response
  opt = get_options("response");
  const bool hasresponse = opt.size();
  std::string responsepath;
  if (hasresponse) {
    if (!rf.empty())
      throw Exception("options -response and -rf are mutually exclusive.");
    responsepath = std::string (opt[0][0]);
  }

  // Get volume indices 
  vector<size_t> idx;
  if (rf.empty() && !hasresponse) {
    idx = shells.largest().get_volumes();
  } else {
    for (size_t k = 0; k < shells.count(); k++)
      idx.insert(idx.end(), shells[k].get_volumes().begin(), shells[k].get_volumes().end());
    std::sort(idx.begin(), idx.end());
  }

  // Select subset
  auto dwisub = Adapter::make <Adapter::Extract1D> (dwi, 3, container_cast<vector<uint32_t>> (idx));

  Eigen::MatrixXf gradsub (idx.size(), grad.cols());
  for (size_t i = 0; i < idx.size(); i++)
    gradsub.row(i) = grad.row(idx[i]).template cast<float>();

  size_t ne = motion.rows()/dwi.size(3);
  Eigen::MatrixXf motionsub (ne * idx.size(), 6);
  for (size_t i = 0; i < idx.size(); i++)
    for (size_t j = 0; j < ne; j++)
      motionsub.row(i * ne + j) = motion.row(idx[i] * ne + j);

  Eigen::MatrixXf Wsub (W.rows(), idx.size());
  for (size_t i = 0; i < idx.size(); i++)
    Wsub.col(i) = W.col(idx[i]);

  Eigen::MatrixXf PEsub;
  if (hasfield) {
    PEsub.resize(idx.size(), PE.cols());
    for (size_t i = 0; i < idx.size(); i++)
      PEsub.row(i) = PE.row(idx[i]);
  }

  // SSP
  DWI::SVR::SSP<float> ssp (DEFAULT_SSPW);
  opt = get_options("ssp");
  if (opt.size()) {
    std::string t = opt[0][0];
    try {
      ssp = DWI::SVR::SSP<float>(std::stof(t));
    } catch (std::invalid_argument& e) {
      try {
        Eigen::VectorXf v = load_vector<float>(t);
        ssp = DWI::SVR::SSP<float>(v);
      } catch (...) {
        throw Exception ("Invalid argument for SSP.");
      }
    }
  }

  // Read voxel weights
  Eigen::VectorXf Wvox = Eigen::VectorXf::Ones(dwisub.size(0)*dwisub.size(1)*dwisub.size(2)*dwisub.size(3));
  opt = get_options("voxweights");
  if (opt.size()) {
    auto voxweights = Image<value_type>::open(opt[0][0]);
    check_dimensions(dwisub, voxweights, 0, 4);
    size_t j = 0;
    for (auto l = Loop("loading voxel weights data", {0, 1, 2, 3})(voxweights); l; l++, j++) {
      Wvox[j] = voxweights.value();
    }
  }

  // Other parameters
  if (rf.empty())
    lmax = get_option_value("lmax", DEFAULT_LMAX);
  else
    lmax = std::min(lmax, (int) get_option_value("lmax", lmax));
  
  float reg = get_option_value("reg", DEFAULT_REG);
  float zreg = get_option_value("zreg", DEFAULT_ZREG);

  value_type tol = get_option_value("tolerance", DEFAULT_TOL);
  size_t maxiter = get_option_value("maxiter", DEFAULT_MAXITER);

  DWI::SVR::QSpaceBasis qbasis {gradsub, lmax, rf, motionsub};

  size_t ncoefs = qbasis.get_ncoefs();
  size_t padding = get_option_value("padding", Math::SH::NforL(lmax));
  if (padding < Math::SH::NforL(lmax))
    throw Exception("user-provided padding too small.");


  // Create source header - needed due to stride handling
  Header srchdr (dwisub);
  Stride::set (srchdr, {1, 2, 3, 4});
  DWI::set_DW_scheme (srchdr, gradsub);
  srchdr.datatype() = DataType::Float32;
  srchdr.sanitise();

  // Create recon header
  Header rechdr (dwisub);
  opt = get_options("template");
  if (opt.size()) {
    rechdr = Header::open(opt[0][0]);
  }
  rechdr.ndim() = 4;
  rechdr.size(3) = ncoefs;
  Stride::set (rechdr, {2, 3, 4, 1});
  rechdr.datatype() = DataType::Float32;
  rechdr.sanitise();

  // Load voxel-wise response onto the reconstruction grid (voxels x shells*orders, x fastest)
  using RowMatrixXf = DWI::SVR::QSpaceBasis::RowMatrixXf;
  std::shared_ptr<RowMatrixXf> response;
  size_t norders = 0;
  if (hasresponse) {
    auto rimg = Image<value_type>::open(responsepath);
    if (rimg.ndim() != 5)
      throw Exception("response image must be 5-D (x, y, z, shell, order).");
    check_dimensions(rechdr, rimg, 0, 3);
    if (size_t (rimg.size(3)) != shells.count())
      throw Exception("no. shells in response image (" + str(rimg.size(3)) + ") does not match the data (" + str(shells.count()) + ").");
    norders = rimg.size(4);
    const size_t nvox = rechdr.size(0)*rechdr.size(1)*rechdr.size(2);
    response = std::make_shared<RowMatrixXf>(nvox, shells.count()*norders);
    size_t v = 0;
    for (auto l = Loop("loading voxel-wise response", {0, 1, 2})(rimg); l; l++, v++)
      for (rimg.index(3) = 0; rimg.index(3) < rimg.size(3); rimg.index(3)++)
        for (rimg.index(4) = 0; rimg.index(4) < rimg.size(4); rimg.index(4)++) {
          float r = rimg.value();
          (*response)(v, rimg.index(3)*norders + rimg.index(4)) = std::isfinite(r) ? r : 0.0f;
        }
    qbasis.set_response(response, shells.count(), norders);
  }

  // Voxels with response estimation and/or the ODF constraint (recon-grid index, x fastest)
  vector<size_t> atomvox;
  {
    auto mopt = get_options("atommask");
    if (mopt.size()) {
      auto amask = Image<bool>::open(mopt[0][0]);
      check_dimensions(rechdr, amask, 0, 3);
      size_t v = 0;
      for (auto l = Loop({0, 1, 2})(amask); l; l++, v++)
        if (amask.value()) atomvox.push_back(v);
    }
  }
  const bool nonneg = get_options("nonneg").size();
  if (nonneg && (!hasresponse || atomvox.empty()))
    throw Exception("-nonneg requires -response and -atommask.");

  // Joint mode: LoRE-SD atoms, initial weights (NNLS projection of the given response onto the atoms)
  opt = get_options("atoms");
  const bool joint = opt.size();
  Eigen::MatrixXd Kr;                         // atoms x shells*orders
  Eigen::MatrixXd Watoms;                     // voxels-with-weights x atoms
  if (joint) {
    if (!hasresponse)
      throw Exception("-atoms requires -response (initial response and the response outside -atommask).");
    Eigen::MatrixXd Kfull = load_matrix<double>(opt[0][0]);
    const size_t nS = shells.count();
    if (Kfull.cols() % nS)
      throw Exception("no. columns of the atom matrix is not a multiple of the no. shells (" + str(nS) + ").");
    const size_t nlK = Kfull.cols() / nS;
    if (nlK < norders)
      throw Exception("atoms have fewer harmonic orders than the response.");
    Kr.resize(Kfull.rows(), nS*norders);
    for (size_t sh = 0; sh < nS; sh++)
      Kr.middleCols(sh*norders, norders) = Kfull.middleCols(sh*nlK, norders);
    if (atomvox.empty())
      throw Exception("-atoms requires -atommask.");
    const Eigen::MatrixXd KKt = Kr * Kr.transpose();
    Watoms.resize(atomvox.size(), Kr.rows());
    DWI::Micro::parallel_for(atomvox.size(), "projecting response onto " + str(Kr.rows()) + " atoms", [&](size_t a) {
      const Eigen::VectorXd r = response->row(atomvox[a]).transpose().cast<double>();
      const Eigen::VectorXd w = DWI::Micro::nnls(KKt, Kr * r);
      Watoms.row(a) = w.transpose();
      response->row(atomvox[a]) = (Kr.transpose() * w).transpose().cast<float>();
    });
  }
  // per-voxel map from reconstruction coefficients to the multi-shell SH series of shell k
  auto to_mssh = [&] (const Eigen::VectorXf& c, size_t voxel, int k, Eigen::Ref<Eigen::VectorXf> sh) {
    if (!hasresponse) {
      sh = qbasis.getShellBasis(k).transpose() * c;
      return;
    }
    const float* r = qbasis.response(voxel, k);
    for (size_t i = 0; i < ncoefs; i++)
      sh[i] = (qbasis.get_band(i) >= 0) ? r[qbasis.get_band(i)] * c[i] : 0.0f;
  };


  // Create mapping
  DWI::SVR::ReconMapping map (rechdr, srchdr, qbasis, motionsub, ssp);

  // Set up scattered data matrix
  INFO("initialise reconstruction matrix");
  DWI::SVR::ReconMatrix R (map, reg, zreg);
  R.setWeights(Wsub);

  R.setVoxelWeights(Wvox);

//  if (hasfield)
//    R.setField(fieldmap, fieldidx, PEsub);




  // Read input data to vector (this enforces positive strides!)
  Eigen::VectorXf y (R.rows()); y.setZero();
  size_t j = 0;
  for (auto lv = Loop("loading image data", {0, 1, 2, 3})(dwisub); lv; lv++, j++) {
      float w = Wsub(size_t(dwisub.index(2)), size_t(dwisub.index(3))) * Wvox[j];
      y[j] = std::sqrt(w) * dwisub.value();
  }

  // Fit scattered data in basis...
  INFO("initialise conjugate gradient solver");

  Eigen::LeastSquaresConjugateGradient<DWI::SVR::ReconMatrix, Eigen::IdentityPreconditioner> cg;
  cg.compute(R);

  cg.setTolerance(tol);
  cg.setMaxIterations(maxiter);

  // Solve y = M x
  Eigen::VectorXf x (R.cols());
  opt = get_options("init");
  if (opt.size()) {
    // load initialisation
    auto init = Image<value_type>::open(opt[0][0]);
    check_dimensions(rechdr, init, 0, 3);
    if ((init.size(3) != shells.count()) || (init.size(4) < Math::SH::NforL(lmax)))
      throw Exception("dimensions of init image don't match.");
    // init vector: least-squares fit of the coefficients to the given multi-shell SH series, voxel by voxel
    const size_t nsh = Math::SH::NforL(lmax);
    Eigen::VectorXf x0 (R.cols()); x0.setZero();
    Eigen::MatrixXf c (shells.count(), nsh);
    Eigen::MatrixXf x2mssh (shells.count()*nsh, ncoefs); x2mssh.setZero();
    for (size_t k = 0; k < shells.count(); k++)
      x2mssh.middleRows(k*nsh, nsh) = qbasis.getShellBasis(k).transpose();
    // pseudo-inverse, computed once
    Eigen::MatrixXf mssh2x = x2mssh.completeOrthogonalDecomposition().pseudoInverse();
    size_t j = 0, voxel = 0;
    // explicit indexing: the previous Loop-based reader left the shell and coefficient indices past their ends on
    // a memory-mapped image and segfaulted intermittently
    ProgressBar progress ("loading initialisation", init.size(2));
    for (ssize_t z = 0; z < init.size(2); z++, ++progress)
    for (ssize_t yy = 0; yy < init.size(1); yy++)
    for (ssize_t xx = 0; xx < init.size(0); xx++, j+=ncoefs, voxel++) {
      init.index(0) = xx; init.index(1) = yy; init.index(2) = z;
      for (ssize_t k = 0; k < init.size(3); k++)
        for (ssize_t i = 0; i < ssize_t (nsh); i++) {
          init.index(3) = k; init.index(4) = i;
          float val = init.value();
          c(k, i) = std::isfinite(val) ? val : 0.0f;
        }
      if (hasresponse) {
        for (size_t i = 0; i < ncoefs; i++) {
          int b = qbasis.get_band(i);
          float num = 0, den = 0;
          if (b >= 0)
            for (size_t k = 0; k < shells.count(); k++) {
              float r = qbasis.response(voxel, k)[b];
              num += r * c(k,i); den += r * r;
            }
          x0[j+i] = (den > 0) ? num / den : 0.0f;
        }
      } else {
        Eigen::VectorXf cv (shells.count()*nsh);
        for (size_t k = 0; k < shells.count(); k++) cv.segment(k*nsh, nsh) = c.row(k).transpose();
        x0.segment(j, ncoefs).noalias() = mssh2x * cv;
      }
    }
    // Warm start as a residual problem: x = x0 + d, with d the least-squares solution of A d = y - A x0 from zero.
    // (Eigen's solveWithGuess evaluates y - A x0 through the operator with a scaling of -1, which the custom
    // ReconMatrix product does not implement: it silently computed y + A x0.)
    INFO("solve from given starting point");
    Eigen::VectorXf Ax0 (R.rows()); Ax0.setZero();
    R.project(Ax0, x0);
    Eigen::VectorXf r = y - Ax0;
    x = x0 + cg.solve(r);
  }
  else {
    INFO("solve from zero starting point");
    x = cg.solve(y);
  }

  CONSOLE("CG: #iterations: " + str(cg.iterations()));
  CONSOLE("CG: estimated error: " + str(cg.error()));

  // ODF non-negativity (as LoRE-SD): min ||y - A c||^2 s.t. Q D^-1 c >= 0 in the voxels of -atommask, by ADMM with
  // splitting q = Q D^-1 c. Q: SH on 129 directions; D = diag(sqrt(4 pi/(2l+1))): the reconstruction coefficients are
  // the LoRE-SD ODF times D and a positive per-voxel scale. Each c-update solves (A^T A + rho P^T P) c = A^T y +
  // rho P^T (z - u) by warm-started conjugate gradients with SHARD's operator; z, u persist across ODF steps.
  Eigen::MatrixXf Pc, PtP;                    // P = Q D^-1 (ndir x ncoefs)
  Eigen::MatrixXf zadmm, uadmm;               // constrained voxels x ndir
  float rho = 0;
  const size_t nadmm = get_option_value("admm", 8), nadmmcg = get_option_value("admm_cg", 4);
  auto ReconAdj = R.adjoint();
  auto odf_nonneg = [&](Eigen::VectorXf& c) {
    const size_t nd = Pc.rows();
    auto Nop = [&](const Eigen::VectorXf& v, Eigen::VectorXf& out) {
      Eigen::VectorXf t (R.rows()); t.setZero(); R.project(t, v);
      out.setZero(R.cols()); ReconAdj.project(out, t);
      for (size_t a = 0; a < atomvox.size(); a++)
        out.segment(atomvox[a]*ncoefs, ncoefs) += rho * (PtP * v.segment(atomvox[a]*ncoefs, ncoefs));
    };
    Eigen::VectorXf Aty (R.cols()); Aty.setZero(); ReconAdj.project(Aty, y);
    if (zadmm.rows() == 0) {
      zadmm.resize(atomvox.size(), nd); uadmm.setZero(atomvox.size(), nd);
      for (size_t a = 0; a < atomvox.size(); a++)
        zadmm.row(a) = (Pc * c.segment(atomvox[a]*ncoefs, ncoefs)).cwiseMax(0.0f).transpose();
    }
    for (size_t k = 0; k < nadmm; k++) {
      Eigen::VectorXf b = Aty;
      for (size_t a = 0; a < atomvox.size(); a++)
        b.segment(atomvox[a]*ncoefs, ncoefs) += rho * (Pc.transpose() * (zadmm.row(a) - uadmm.row(a)).transpose());
      // conjugate gradients on the normal equations, warm start from c
      Eigen::VectorXf Nc; Nop(c, Nc);
      Eigen::VectorXf r = b - Nc, p = r, Np;
      double rr = r.squaredNorm();
      for (size_t it = 0; it < nadmmcg && rr > 0; it++) {
        Nop(p, Np);
        const double alpha = rr / double(p.dot(Np));
        c += float(alpha) * p; r -= float(alpha) * Np;
        const double rr2 = r.squaredNorm();
        p = r + float(rr2 / rr) * p; rr = rr2;
      }
      size_t neg = 0;
      for (size_t a = 0; a < atomvox.size(); a++) {
        const Eigen::VectorXf q = Pc * c.segment(atomvox[a]*ncoefs, ncoefs);
        const Eigen::VectorXf zn = (q + uadmm.row(a).transpose()).cwiseMax(0.0f);
        uadmm.row(a) += (q - zn).transpose();
        zadmm.row(a) = zn.transpose();
        neg += (q.array() < -1e-3f * std::max(1e-12f, q.cwiseAbs().maxCoeff())).count();
      }
      INFO("ADMM " + str(k+1) + "/" + str(nadmm) + ": " + str(100.0 * neg / (atomvox.size() * nd)) + " % of constraints violated");
    }
  };
  if (nonneg) {
    const Eigen::MatrixXd dirs = DWI::Directions::tesselation_129();
    const Eigen::MatrixXd Q = Math::SH::init_transform(dirs, lmax);
    Pc = Q.cast<float>();
    for (size_t i = 0; i < ncoefs; i++) {
      int l = 0; while (Math::SH::NforL(l) <= i) l += 2;
      Pc.col(i) /= std::sqrt(4.0f * Math::pi / (2*l+1));
    }
    PtP = Pc.transpose() * Pc;
    // rho: balance the constraint term against the data term (Hutchinson estimate of the mean diagonal of A^T A)
    Eigen::VectorXf probe = Eigen::VectorXf::Random(R.cols()).array().sign(), t (R.rows()); t.setZero();
    R.project(t, probe);
    rho = float(t.squaredNorm() / R.cols()) / (PtP.trace() / ncoefs);
    INFO("ODF non-negativity: rho = " + str(rho));
    odf_nonneg(x);
  }

  // Joint SHARD / LoRE-SD: alternate response (atom weights, ODF fixed) and ODF (response fixed) steps
  if (joint) {
    const size_t nS = shells.count(), nunk = nS*norders, nvox = rechdr.size(0)*rechdr.size(1)*rechdr.size(2);
    const size_t nalt = get_option_value("alternate", 2), nsteps = get_option_value("atomsteps", 3);
    const size_t nrows = map.rows(), nvol = gradsub.rows(), nshot = motionsub.rows() / nvol;
    // response-step mapping: ODF fixed, coefficients R(voxel, shell, order)
    auto odfmat = std::make_shared<RowMatrixXf>(nvox, ncoefs);
    DWI::SVR::QSpaceBasis rbasis {gradsub, lmax, rf, motionsub};
    rbasis.set_odf(odfmat, nS, norders);
    Header rhdr (rechdr);
    rhdr.size(3) = nunk;
    Stride::set (rhdr, {2, 3, 4, 1});
    rhdr.sanitise();
    DWI::SVR::ReconMapping rmap (rhdr, srchdr, rbasis, motionsub, ssp);
    Eigen::VectorXf sqw (nrows);
    {
      Eigen::VectorXf ones = Eigen::VectorXf::Ones(nrows);
      ImageView<float> src (srchdr, ones.data());
      size_t jj = 0;
      for (auto l = Loop()(src); l; l++, jj++)
        sqw[jj] = std::sqrt(Wsub((size_t) src.index(2), (size_t) src.index(3)) * Wvox[jj]);
    }
    Eigen::Map<Eigen::VectorXf> Rv (response->data(), response->size());
    const Eigen::VectorXf yd = y.head(nrows);
    auto forward = [&](const Eigen::VectorXf& Rvec, Eigen::VectorXf& pred) {
      Eigen::VectorXf copy = Rvec; pred.setZero(nrows);
      ImageView<float> rv (rhdr, copy.data()); ImageView<float> src (srchdr, pred.data());
      rmap.x2y(rv, src);
      pred.array() *= sqw.array();
    };
    auto adjoint = [&](const Eigen::VectorXf& res, Eigen::VectorXf& g) {
      Eigen::VectorXf copy = res.array() * sqw.array(); g.setZero(nvox*nunk);
      ImageView<float> gv (rhdr, g.data()); ImageView<float> src (srchdr, copy.data());
      rmap.y2x(gv, src);
    };
    // per-volume weight (mean slice weight) and the projection row of the volume's first shot
    Eigen::VectorXd wvol (nvol);
    for (size_t v = 0; v < nvol; v++) wvol[v] = Wsub.col(v).mean();
    vector<size_t> volshell (nvol);
    for (size_t v = 0; v < nvol; v++) volshell[v] = qbasis.get_shell(v*nshot);
    const size_t K = Kr.rows();
    Eigen::VectorXf pred;
    if (getenv("DWIRECON_CHECK_ADJOINT")) {   // dot-product test of the response-step operator
      std::copy(x.data(), x.data() + x.size(), odfmat->data());
      Eigen::VectorXf r1 = Eigen::VectorXf::Random(nvox*nunk), y1 = Eigen::VectorXf::Random(nrows), Ar, Aty;
      forward(r1, Ar); adjoint(y1, Aty);
      CONSOLE("adjoint check: <A r, y> = " + str(double(Ar.dot(y1))) + ", <r, A^T y> = " + str(double(r1.dot(Aty))));
    }
    for (size_t it = 0; it < nalt; it++) {
      // fix the ODF
      std::copy(x.data(), x.data() + x.size(), odfmat->data());
      forward(Rv, pred);
      double f0 = 0.5 * (yd - pred).squaredNorm();
      const double fstart = f0;
      for (size_t st = 0; st < nsteps; st++) {
        Eigen::VectorXf g;
        adjoint(yd - pred, g);                       // g = A^T W (y - A R) = -gradient wrt R
        Eigen::MatrixXd Wnew (Watoms.rows(), K);
        DWI::Micro::parallel_for(atomvox.size(), "response step " + str(it+1) + "." + str(st+1), [&](size_t a) {
          const size_t vox = atomvox[a];
          const float* c = odfmat->data() + vox*ncoefs;
          // local quadratic model: H = sum_v w_v a_v a_v^T, a_v = K_shell(v) d_v, d_v,l = sum_{band l} Y_i(u_v) c_i
          Eigen::MatrixXd B (nvol, K);
          for (size_t v = 0; v < nvol; v++) {
            Eigen::Ref<const Eigen::VectorXf> q = qbasis.get_projection(v*nshot);
            Eigen::VectorXd d = Eigen::VectorXd::Zero(norders);
            for (size_t i = 0; i < ncoefs; i++)
              if (qbasis.get_band(i) >= 0) d[qbasis.get_band(i)] += double(q[i]) * c[i];
            B.row(v) = std::sqrt(wvol[v]) * (Kr.middleCols(volshell[v]*norders, norders) * d).transpose();
          }
          Eigen::MatrixXd H = B.transpose() * B;
          H.diagonal().array() += 1e-6 * std::max(H.trace() / K, 1e-12);
          const Eigen::VectorXd grad = -(Kr * g.segment(vox*nunk, nunk).cast<double>());
          const Eigen::VectorXd wk = Watoms.row(a).transpose();
          Wnew.row(a) = DWI::Micro::nnls(H, H * wk - grad).transpose();
        });
        // line search on the true objective along w_k + alpha (w* - w_k) (stays non-negative)
        bool accepted = false;
        Eigen::VectorXf Rtry = Rv;
        for (double alpha = 1.0; alpha > 0.06; alpha *= 0.5) {
          for (size_t a = 0; a < atomvox.size(); a++) {
            const Eigen::VectorXd w = Watoms.row(a).transpose() + alpha * (Wnew.row(a) - Watoms.row(a)).transpose();
            Rtry.segment(atomvox[a]*nunk, nunk) = (Kr.transpose() * w).cast<float>();
          }
          Eigen::VectorXf ptry;
          forward(Rtry, ptry);
          const double f = 0.5 * (yd - ptry).squaredNorm();
          if (f < f0) {
            for (size_t a = 0; a < atomvox.size(); a++)
              Watoms.row(a) += alpha * (Wnew.row(a) - Watoms.row(a));
            Rv = Rtry; pred = ptry; f0 = f; accepted = true;
            INFO("response step " + str(it+1) + "." + str(st+1) + ": step " + str(alpha) + ", misfit " + str(f));
            break;
          }
        }
        if (!accepted) break;
      }
      // fix the scale ambiguity: max_shell |R(shell, l=0)| = 1 per voxel, the ODF takes the scale
      for (size_t a = 0; a < atomvox.size(); a++) {
        const size_t vox = atomvox[a];
        float sc = 0;
        for (size_t sh = 0; sh < nS; sh++) sc = std::max(sc, std::abs((*response)(vox, sh*norders)));
        if (sc > 0) {
          response->row(vox) /= sc; Watoms.row(a) /= sc;
          x.segment(vox*ncoefs, ncoefs) *= sc;
          if (zadmm.rows()) { zadmm.row(a) *= sc; uadmm.row(a) *= sc; }   // same voxel list (atommask)
        }
      }
      // ODF step with the new response (warm start)
      Eigen::VectorXf Ax0 (R.rows()); Ax0.setZero();
      R.project(Ax0, x);
      Eigen::VectorXf r = y - Ax0;
      if (nonneg) odf_nonneg(x);
      else x += cg.solve(r);
      Eigen::VectorXf pc; std::copy(x.data(), x.data() + x.size(), odfmat->data()); forward(Rv, pc);
      CONSOLE("joint " + str(it+1) + "/" + str(nalt) + ": data misfit " + str(fstart) + " -> " + str(f0) +
              " (response) -> " + str(0.5 * (yd - pc).squaredNorm()) + " (ODF)");
    }
  }


  // Write result to output file
  Header msshhdr (rechdr);
  msshhdr.ndim() = 5;
  msshhdr.size(3) = shells.count();
  msshhdr.size(4) = padding;
  Stride::set_from_command_line (msshhdr, {3, 4, 5, 2, 1});
  msshhdr.datatype() = DataType::from_command_line (DataType::Float32);
  Metadata::PhaseEncoding::clear_scheme (msshhdr.keyval());
  // store b-values and counts
  {
  std::stringstream ss;
  for (auto b : shells.get_bvalues())
    ss << b << ",";
  std::string key = "shells";
  std::string val = ss.str(); val.erase(val.length()-1);
  msshhdr.keyval()[key] = val;
  } {
  std::stringstream ss;
  for (auto c : shells.get_counts())
    ss << c << ",";
  std::string key = "shellcounts";
  std::string val = ss.str(); val.erase(val.length()-1);
  msshhdr.keyval()[key] = val;
  }

  auto out = Image<value_type>::create (argument[1], msshhdr);

  j = 0;
  size_t voxel = 0;
  Eigen::VectorXf c (ncoefs);
  Eigen::VectorXf sh (padding); sh.setZero();
  for (auto l = Loop("writing result to image", {0, 1, 2})(out); l; l++, j+=ncoefs, voxel++) {
    c = x.segment(j, ncoefs);
    for (size_t k = 0; k < shells.count(); k++) {
      out.index(3) = k;
      auto head = sh.head(Math::SH::NforL(lmax));
      to_mssh(c, voxel, k, head);
      out.row(4) = sh;
    }
  }

  // Output the per-voxel spherical function itself (the ODF up to scale) when a response was given
  opt = get_options("odf");
  if (opt.size()) {
    if (!hasresponse)
      throw Exception("option -odf requires -response.");
    Header odfhdr (rechdr);
    Stride::set (odfhdr, {2, 3, 4, 1});
    Metadata::PhaseEncoding::clear_scheme (odfhdr.keyval());
    auto odf = Image<value_type>::create(opt[0][0], odfhdr);
    j = 0;
    for (auto l = Loop("writing ODF", {0, 1, 2})(odf); l; l++, j+=ncoefs)
      odf.row(3) = x.segment(j, ncoefs);
  }


  // Output source prediction
  bool complete = get_options("complete").size();
  opt = get_options("spred");
  if (opt.size()) {
    srchdr.size(3) = (complete) ? dwi.size(3) : dwisub.size(3);
    auto spred = Image<value_type>::create(opt[0][0], srchdr);
    auto recon = ImageView<value_type>(rechdr, x.data());
    map.x2y(recon, spred);
  }



  // Output the voxel-wise response
  opt = get_options("response_out");
  if (opt.size()) {
    if (!hasresponse)
      throw Exception("-response_out requires -response.");
    Header rh (rechdr);
    rh.ndim() = 5; rh.size(3) = shells.count(); rh.size(4) = norders;
    Stride::set (rh, {3, 4, 5, 2, 1});
    Metadata::PhaseEncoding::clear_scheme (rh.keyval());
    rh.keyval().erase("dw_scheme");
    auto rout = Image<value_type>::create(opt[0][0], rh);
    size_t vox = 0;
    for (auto l = Loop("writing response", {0, 1, 2})(rout); l; l++, vox++)
      for (rout.index(3) = 0; rout.index(3) < rout.size(3); rout.index(3)++)
        for (rout.index(4) = 0; rout.index(4) < rout.size(4); rout.index(4)++)
          rout.value() = (*response)(vox, rout.index(3)*norders + rout.index(4));
  }
}
