/* Copyright (c) 2017-2019 Daan Christiaens
 *
 * MRtrix and this add-on module are distributed in the hope
 * that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.
 */

#include <algorithm>
#include <sstream>

#include "command.h"
#include "image.h"
#include "math/SH.h"
#include "dwi/gradient.h"
#include "metadata/phase_encoding.h"
#include "dwi/shells.h"
#include "dwi/svr/shells.h"
#include "adapter/extract.h"

#include "dwi/svr/qspacebasis.h"
#include "dwi/svr/mapping.h"
#include "dwi/svr/recon.h"

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
  + Argument ("DWI", "the input DWI image header.").type_image_in()
  + Argument ("SH", "the input spherical harmonics coefficients image.").type_image_in()
  + Argument ("spred", "source prediction output.").type_image_out();


  OPTIONS
  + Option ("motion", "The motion parameters associated with input slices or volumes. "
                      "These are supplied as a matrix of 6 columns encoding the rigid "
                      "transformations w.r.t. scanner space in se(3) Lie algebra." )
    + Argument ("file").type_file_in()

  + Option ("rf", "Basis functions for the radial (multi-shell) domain, provided as matrices in which "
                  "rows correspond with shells and columns with SH harmonic bands.").allow_multiple()
    + Argument ("b").type_file_in()
  
  + Option ("lmax", "The maximum harmonic order for the output series. (default = " + str(DEFAULT_LMAX) + ")")
    + Argument ("order").type_integer(0, 30)

  + Option ("weights", "Slice weights, provided as a matrix of dimensions Nslices x Nvols.")
    + Argument ("W").type_file_in()

  + Option ("ssp", "Slice sensitivity profile, either as text file or as a scalar slice thickness for a "
                   "Gaussian SSP, relative to the voxel size. (default = " + str(DEFAULT_SSPW)  + ")")
    + Argument ("w").type_text()

  + Option ("field", "Static susceptibility field, aligned in recon space.")
    + Argument ("map").type_image_in()
    + Argument ("idx").type_integer()

  + DWI::GradImportOptions()

  + Metadata::PhaseEncoding::ImportOptions;

}


typedef float value_type;



void run ()
{
  // Load input data
  auto dwi = Image<value_type>::open(argument[0]);

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
      throw Exception("Weights marix dimensions don't match image dimensions.");
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

  // Get volume indices 
  vector<size_t> idx;
  if (rf.empty()) {
    idx = shells.largest().get_volumes();
  } else {
    for (size_t k = 0; k < shells.count(); k++)
      idx.insert(idx.end(), shells[k].get_volumes().begin(), shells[k].get_volumes().end());
    std::sort(idx.begin(), idx.end());
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

  // Other parameters
  if (rf.empty())
    lmax = get_option_value("lmax", DEFAULT_LMAX);
  else
    lmax = std::min(lmax, (int) get_option_value("lmax", lmax));
  
  DWI::SVR::QSpaceBasis qbasis {grad.cast<float>(), lmax, rf, motion};


    auto init = Image<value_type>::open(argument[1]);
    check_dimensions(dwi, init, 0, 3);
    if ((init.size(3) != shells.count()) || (init.size(4) < Math::SH::NforL(lmax)))
      throw Exception("dimensions of init image don't match.");

    // coefficients held as in dwirecon: a vector with the coefficient axis fastest, viewed through ImageView
    // (a scratch Image adapted by QSpaceMapping segfaults in the forward projection with current MRtrix)
    size_t ncoefs = qbasis.get_ncoefs();
    Header rechdr (init);
    rechdr.ndim() = 4;
    rechdr.size(3) = ncoefs;
    Stride::set (rechdr, {2, 3, 4, 1});
    rechdr.datatype() = DataType::Float32;
    rechdr.sanitise();
    Eigen::VectorXf x (voxel_count(rechdr));

    // convert from mssh
    const size_t nsh = Math::SH::NforL(lmax);
    Eigen::VectorXf c (shells.count() * nsh);
    Eigen::MatrixXf x2mssh (c.size(), ncoefs); x2mssh.setZero();
    for (size_t k = 0; k < shells.count(); k++)
      x2mssh.middleRows(k*nsh, nsh) = qbasis.getShellBasis(k).transpose();
    auto mssh2x = x2mssh.fullPivHouseholderQr();
    size_t j = 0;
    for (auto l = Loop("loading SH coefficients", {0, 1, 2}) (init); l; l++, j += ncoefs) {
      for (init.index(3) = 0; init.index(3) < init.size(3); init.index(3)++)
        for (init.index(4) = 0; init.index(4) < ssize_t (nsh); init.index(4)++) {
          float val = init.value();
          c[init.index(3)*nsh + init.index(4)] = std::isfinite(val) ? val : 0.0f;
        }
      x.segment(j, ncoefs) = mssh2x.solve(c);
    }

    Header srchdr (dwi);
    Stride::set (srchdr, {1, 2, 3, 4});
    DWI::set_DW_scheme (srchdr, grad);
    srchdr.datatype() = DataType::Float32;
    srchdr.sanitise();

    DWI::SVR::ReconMapping map (rechdr, srchdr, qbasis, motion, ssp);
    ImageView<value_type> recon (rechdr, x.data());

    auto spred = Image<value_type>::create(argument[2], srchdr);

    map.x2y(recon, spred);



}




