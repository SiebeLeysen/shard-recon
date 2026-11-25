/* Copyright (c) 2017-2019 Daan Christiaens
 *
 * MRtrix and this add-on module are distributed in the hope
 * that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.
 */

#ifndef __dwi_svr_qspacebasis_h__
#define __dwi_svr_qspacebasis_h__


#include <atomic>
#include <algorithm>
#include <memory>
#include <Eigen/Dense>

#include "types.h"
#include "math/SH.h"
#include "dwi/shells.h"
#include "dwi/svr/shells.h"
#include "adapter/base.h"
#include "algo/loop.h"

#include "dwi/svr/param.h"


namespace MR
{
  namespace Adapter
  {
    template <class ImageType>
    class ReadCache : public Adapter::Base<ReadCache<ImageType>, ImageType>
    {
      public:
        using base_type = Adapter::Base<ReadCache<ImageType>, ImageType>;
        using value_type = typename ImageType::value_type;

        using base_type::parent;

        ReadCache (const ImageType& parent)
          : base_type (parent)
        {
          Header hdr (parent);
          buffer = Image<value_type>::scratch(hdr, "temporary buffer");
        }

        ReadCache (const ReadCache& other)
          : ReadCache (other.parent())
        { }

        FORCE_INLINE ssize_t get_index (size_t axis) const {
          return buffer.get_index(axis);
        }
        FORCE_INLINE void move_index (size_t axis, ssize_t increment) {
          buffer.move_index(axis, increment);
        }
        FORCE_INLINE void reset () {
          buffer.reset();
        }

        void flush () {
          // clear buffer
          reset();
          std::fill_n(buffer.address(), voxel_count(buffer), value_type(NAN));
        }

        FORCE_INLINE value_type value () {
          value_type val = *buffer.address();
          if (!std::isfinite(val)) load(val);
          return val;
        }

        FORCE_INLINE void set_shotidx (size_t idx) {
          flush();
          parent().set_shotidx(idx);
        }

      private:
        Image<value_type> buffer;

        FORCE_INLINE void load (value_type& val) {
          assign_pos_of (buffer).to (parent());
          *buffer.address() = val = parent().value();
        }
    };

    template <class ImageType>
    class WriteCache : public Adapter::Base<WriteCache<ImageType>, ImageType>
    {
      public:
        using base_type = Adapter::Base<WriteCache<ImageType>, ImageType>;
        using value_type = typename ImageType::value_type;

        using base_type::parent;

        WriteCache (const ImageType& parent)
          : base_type (parent)
        {
          Header hdr (parent);
          buffer = Image<value_type>::scratch(hdr, "temporary buffer");
          // initialise lock image
          static_assert (sizeof(std::atomic_flag) == sizeof(uint8_t), "std::atomic_flag expected to be 1 byte");
          lock = Image<uint8_t>::scratch(hdr, "temporary buffer lock");
        }

        WriteCache (const WriteCache& other)
          : base_type (other.parent()), lock (other.lock)
        {
          Header hdr (other.parent());
          buffer = Image<value_type>::scratch(hdr, "temporary buffer");
        }

        FORCE_INLINE ssize_t get_index (size_t axis) const {
          return buffer.get_index(axis);
        }
        FORCE_INLINE void move_index (size_t axis, ssize_t increment) {
          buffer.move_index(axis, increment);
        }
        FORCE_INLINE void reset () {
          buffer.reset();
        }

        void flush () {
          // delayed write back
          for (auto l = Loop() (buffer); l; l++) {
            if (buffer.value()) { assign_pos_of (buffer).to (parent(), lock);
              std::atomic_flag* flag = reinterpret_cast<std::atomic_flag*> (lock.address());
              while (flag->test_and_set(std::memory_order_acquire)) ;
              parent().adjoint_add(buffer.value());
              flag->clear(std::memory_order_release);
            }
          }
          // clear buffer
          reset();
          std::fill_n(buffer.address(), voxel_count(buffer), value_type(0));
        }

        FORCE_INLINE void adjoint_add (value_type val) {
          *buffer.address() += val;
        }

        FORCE_INLINE void set_shotidx (size_t idx) {
          flush();
          parent().set_shotidx(idx);
        }

      private:
        Image<value_type> buffer;
        Image<uint8_t> lock;
    };

    template <template <class ImageType> class AdapterType, class ImageType, typename... Args>
    inline ReadCache<AdapterType<ImageType>> makecached (const ImageType& parent, Args&&... args) {
      return { { parent, std::forward<Args> (args)... } };
    }

    template <template <class ImageType> class AdapterType, class ImageType, typename... Args>
    inline WriteCache<AdapterType<ImageType>> makecached_add (const ImageType& parent, Args&&... args) {
      return { { parent, std::forward<Args> (args)... } };
    }

  }

  namespace DWI
  {
    namespace SVR
    {

      class QSpaceBasis
      {
        public:
          typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXf;

          QSpaceBasis (const Eigen::MatrixXf& grad, const int lmax,
                       const vector<Eigen::MatrixXf>& rf, const Eigen::MatrixXf& rigid)
            : lmax (lmax), nv (grad.rows()), ne (rigid.rows() / nv), nc (get_ncoefs(rf)), rf_empty (rf.empty()),
              shellbasis (init_shellbasis(grad, rf))
          {
            init_Y(grad, rigid);
          }

          QSpaceBasis (const Eigen::MatrixXf& grad, const int lmax,
                       const vector<Eigen::MatrixXf>& rf)
            : lmax (lmax), nv (grad.rows()), ne (1), nc (get_ncoefs(rf)), rf_empty (rf.empty()),
              shellbasis (init_shellbasis(grad, rf))
          {
            RowMatrixXf zero (nv, 6);
            init_Y(grad, zero);
          }

          FORCE_INLINE Eigen::Ref<const Eigen::VectorXf> get_projection (size_t idx) const { return Y.row(idx); }
          FORCE_INLINE Eigen::Ref<const Eigen::VectorXf> get_projection (size_t v, size_t z) const { return Y.row(v*ne + z); }

          FORCE_INLINE size_t get_ncoefs () const { return nc; }
          FORCE_INLINE size_t get_shell (size_t idx) const { return rowshell[idx]; }

          // Voxel-wise response (LoRE-SD factorisation). The coefficients are then the SH coefficients of one
          // spherical function per voxel (the ODF, up to scale), and the signal of shell s in voxel j is
          //   S_s(u) = sum_l R(j, s, l) sum_m c_lm Y_lm(u),
          // i.e. the per-shell SH series is the voxel's response times the ODF, order by order. R is stored as a
          // (voxels x shells*orders) row-major matrix, voxels in recon-grid order (x fastest), orders l = 0, 2, 4, ...
          void set_response (std::shared_ptr<const RowMatrixXf> R, size_t nshells, size_t norders)
          {
            if (!rf_empty)
              throw Exception ("a voxel-wise response cannot be combined with a radial basis (-rf).");
            if (R->cols() != ssize_t (nshells * norders))
              throw Exception ("voxel-wise response has wrong dimensions.");
            resp = R; nsh = nshells; nl = norders;
            band.resize (nc);
            for (int l = 0; l <= lmax; l += 2)
              for (size_t i = Math::SH::NforL (l-2); i < Math::SH::NforL (l); i++)
                band[i] = (l/2 < int (nl)) ? l/2 : -1;    // orders beyond the response are not represented: 0
          }
          // Response as the unknown (joint SHARD / LoRE-SD fit, response step): the ODF is fixed per voxel (C: voxels x
          // ncoefs, recon-grid order) and the mapping's coefficients are R(j, shell, l), shells*orders per voxel:
          //   S_s(u) = sum_l R(j, s, l) d_l(u),   d_l(u) = sum_m c_lm(j) Y_lm(u)
          void set_odf (std::shared_ptr<const RowMatrixXf> C, size_t nshells, size_t norders)
          {
            if (!rf_empty)
              throw Exception ("the response step cannot be combined with a radial basis (-rf).");
            if (C->cols() != ssize_t (nc))
              throw Exception ("fixed ODF has wrong number of coefficients.");
            odf = C; nsh = nshells; nl = norders;
            band.resize (nc);
            for (int l = 0; l <= lmax; l += 2)
              for (size_t i = Math::SH::NforL (l-2); i < Math::SH::NforL (l); i++)
                band[i] = (l/2 < int (nl)) ? l/2 : -1;
          }
          FORCE_INLINE bool response_unknown () const { return bool (odf); }
          FORCE_INLINE const float* odf_of (size_t voxel) const { return odf->data() + voxel*odf->cols(); }
          // number of coefficients per voxel of the image this basis maps from
          FORCE_INLINE size_t get_nunknowns () const { return odf ? nsh*nl : nc; }
          FORCE_INLINE bool has_response () const { return bool (resp); }
          FORCE_INLINE const float* response (size_t voxel, size_t shell) const { return resp->data() + voxel*resp->cols() + shell*nl; }
          FORCE_INLINE int get_band (size_t i) const { return band[i]; }
          FORCE_INLINE size_t response_orders () const { return nl; }
          FORCE_INLINE const RowMatrixXf& getY () const { return Y; }
          FORCE_INLINE const Eigen::MatrixXf& getShellBasis (const int shellidx) const { return shellbasis[shellidx]; }

        private:
          const int lmax;
          const size_t nv, ne, nc;
          const bool rf_empty;
          const vector<Eigen::MatrixXf> shellbasis;
          RowMatrixXf Y;
          vector<size_t> rowshell;
          std::shared_ptr<const RowMatrixXf> resp, odf;
          size_t nsh = 0, nl = 0;
          vector<int> band;

          vector<Eigen::MatrixXf> init_shellbasis(const Eigen::MatrixXf& grad, const vector<Eigen::MatrixXf>& rf) const
          {
            SVR::SVRShells shells (grad.template cast<double>());
            vector<Eigen::MatrixXf> basis;

            for (size_t s = 0; s < shells.count(); s++) {
              Eigen::MatrixXf B;
              if (rf.empty()) {
                B.setIdentity(Math::SH::NforL(lmax), Math::SH::NforL(lmax));
              }
              else {
                B.setZero(nc, Math::SH::NforL(lmax));
                size_t j = 0;
                for (auto& r : rf) {
                  for (size_t l = 0; l < r.cols() and 2*l <= lmax; l++)
                    for (size_t i = l*(2*l-1); i < (l+1)*(2*l+1); i++, j++)
                      B(j,i) = r(s,l);
                }
              }
              basis.push_back(B);
            }

            return basis;
          }

          void init_Y(const Eigen::MatrixXf& grad, const Eigen::MatrixXf& motion)
          {
            DEBUG("initialise Y");
            assert (grad.rows() == nv);     // one gradient per volume

            vector<size_t> idx = get_shellidx(grad);
            Y.resize(nv*ne, nc);
            rowshell.resize(nv*ne);

            Eigen::Vector3f vec;
            Eigen::Matrix3f rot;
            rot.setIdentity();
            Eigen::VectorXf delta;

            for (size_t v = 0; v < nv; v++) {
              vec = {grad(v, 0), grad(v, 1), grad(v, 2)};
              for (size_t e = 0; e < ne; e++) {
                // rotate vector with motion parameters
                rot = get_rotation(motion.row(v*ne+e));
                // evaluate basis functions
                Math::SH::delta(delta, rot*vec, lmax);
                Y.row(v*ne+e) = shellbasis[idx[v]]*delta;
                rowshell[v*ne+e] = idx[v];
              }
            }

          }

          inline Eigen::Matrix3f get_rotation(const Eigen::VectorXf& p) const
          {
            Eigen::Matrix3f m = se3exp(p).topLeftCorner<3,3>();
            return m;
          }

          inline size_t get_ncoefs(const vector<Eigen::MatrixXf>& rf) const
          {
            size_t n = 0;
            if (rf.empty()) {
              n = Math::SH::NforL(lmax);
            } else {
              for (auto& r : rf)
                n += Math::SH::NforL(std::min(2*(int(r.cols())-1), lmax));
            }
            return n;
          }

          vector<size_t> get_shellidx(const Eigen::MatrixXf& grad) const
          {
            SVR::SVRShells shells (grad.template cast<double>());
            vector<size_t> idx (shells.volumecount());

            for (size_t s = 0; s < shells.count(); s++) {
              for (auto v : shells[s].get_volumes())
                idx[v] = s;
            }
            return idx;
          }

      };


      template <class ImageType>
      class QSpaceMapping : public Adapter::Base<QSpaceMapping<ImageType>, ImageType>
      {
        public:
          using base_type = Adapter::Base<QSpaceMapping<ImageType>, ImageType>;
          using value_type = typename ImageType::value_type;
          using vector_type = typename Eigen::Matrix<value_type, Eigen::Dynamic, 1>;

          using base_type::parent;

          QSpaceMapping (const ImageType& parent, const QSpaceBasis& basis)
            : base_type (parent), basis (basis)
          {
            assert (parent.ndim() == 4);
            assert (parent.size(3) == basis.get_nunknowns());
            assert (parent.stride(3) == 1);
            set_shotidx(0);
          }

          FORCE_INLINE size_t ndim () const { return 3; }

          FORCE_INLINE ssize_t get_index (size_t axis) const { return parent().get_index (axis); }
          FORCE_INLINE void move_index (size_t axis, ssize_t increment) { parent().move_index (axis, increment); }

          FORCE_INLINE value_type value () const {
            assert (parent().index(3) == 0);
            if (basis.response_unknown()) {
              const value_type* R = parent().address() + shell*basis.response_orders();
              value_type d[16]; bands (d);
              value_type s = 0;
              for (size_t l = 0; l < basis.response_orders(); l++) s += R[l] * d[l];
              return s;
            }
            Eigen::Map<vector_type> c = {parent().address(), qr.size()};
            if (!basis.has_response())
              return qr.dot(c);
            const float* r = basis.response (voxel(), shell);
            value_type s = 0;
            for (ssize_t i = 0; i < qr.size(); i++)
              if (basis.get_band(i) >= 0) s += r[basis.get_band(i)] * qr[i] * c[i];
            return s;
          }

          FORCE_INLINE void adjoint_add (value_type val) {
            assert (parent().index(3) == 0);
            if (basis.response_unknown()) {
              value_type* R = parent().address() + shell*basis.response_orders();
              value_type d[16]; bands (d);
              for (size_t l = 0; l < basis.response_orders(); l++) R[l] += val * d[l];
              return;
            }
            Eigen::Map<vector_type> c = {parent().address(), qr.size()};
            if (!basis.has_response()) {
              c += val * qr;
              return;
            }
            const float* r = basis.response (voxel(), shell);
            for (ssize_t i = 0; i < qr.size(); i++)
              if (basis.get_band(i) >= 0) c[i] += val * r[basis.get_band(i)] * qr[i];
          }

          FORCE_INLINE void set_shotidx (size_t idx) {
            qr = basis.get_projection(idx);
            shell = basis.get_shell(idx);
          }

        private:
          const QSpaceBasis& basis;
          vector_type qr;
          size_t shell = 0;

          // d_l = sum over band l of Y_lm(u) c_lm(voxel), with the fixed ODF of this voxel
          FORCE_INLINE void bands (value_type* d) const {
            const float* c = basis.odf_of (voxel());
            for (size_t l = 0; l < basis.response_orders(); l++) d[l] = 0;
            for (ssize_t i = 0; i < qr.size(); i++)
              if (basis.get_band(i) >= 0) d[basis.get_band(i)] += qr[i] * c[i];
          }

          FORCE_INLINE size_t voxel () const {   // linear voxel index in the recon grid, x fastest
            return size_t (parent().index(0)) + size_t (parent().size(0)) *
                   (size_t (parent().index(1)) + size_t (parent().size(1)) * size_t (parent().index(2)));
          }
      };


    }
  }

}


#endif
