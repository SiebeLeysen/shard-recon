/* Shell grouping for SHARD without a minimum shell size.
 *
 * MRtrix's DWI::Shells clusters b-values with a minimum linkage of DWI_SHELLS_MIN_LINKAGE (3) volumes, so shells
 * with only one or two volumes cannot be classified. In multi-dimensional acquisitions (several b-tensor shapes and
 * echo times, each given its own placeholder b-value), and in subsampled protocols, such small shells are common.
 * SVRShells groups volumes whose b-values lie within DWI_SHELLS_EPSILON of a shell's first volume, keeps every
 * group regardless of size, and orders the shells by mean b-value (as DWI::Shells does). Shell selection options
 * are not supported.
 */

#ifndef __dwi_svr_shells_h__
#define __dwi_svr_shells_h__

#include <algorithm>
#include <numeric>
#include "types.h"
#include "dwi/shells.h"

namespace MR
{
  namespace DWI
  {
    namespace SVR
    {

      class SVRShell
      {
        public:
          SVRShell (const vector<size_t>& volumes, default_type mean) : volumes (volumes), mean (mean) { }
          const vector<size_t>& get_volumes () const { return volumes; }
          size_t count () const { return volumes.size(); }
          default_type get_mean () const { return mean; }
        private:
          vector<size_t> volumes;
          default_type mean;
      };


      class SVRShells
      {
        public:
          template <class MatrixType>
          SVRShells (const MatrixType& grad)
          {
            const size_t nv = grad.rows();
            vector<bool> assigned (nv, false);
            vector<SVRShell> found;
            for (size_t v = 0; v < nv; v++) {
              if (assigned[v]) continue;
              const default_type b0 = grad(v,3);
              vector<size_t> vols;
              default_type sum = 0.0;
              for (size_t w = v; w < nv; w++)
                if (!assigned[w] && std::abs (default_type (grad(w,3)) - b0) < DWI_SHELLS_EPSILON) {
                  assigned[w] = true; vols.push_back (w); sum += grad(w,3);
                }
              found.emplace_back (vols, sum / vols.size());
            }
            std::sort (found.begin(), found.end(), [](const SVRShell& a, const SVRShell& b) { return a.get_mean() < b.get_mean(); });
            shells = found;
            nvol = nv;
          }

          void select_shells (bool, bool, bool) { }     // selection not supported: all shells are used
          size_t count () const { return shells.size(); }
          size_t volumecount () const { return nvol; }
          const SVRShell& operator[] (size_t i) const { return shells[i]; }
          const SVRShell& largest () const {
            return *std::max_element (shells.begin(), shells.end(), [](const SVRShell& a, const SVRShell& b) { return a.count() < b.count(); });
          }
          vector<default_type> get_bvalues () const { vector<default_type> b; for (auto& s : shells) b.push_back (s.get_mean()); return b; }
          vector<size_t> get_counts () const { vector<size_t> c; for (auto& s : shells) c.push_back (s.count()); return c; }

        private:
          vector<SVRShell> shells;
          size_t nvol = 0;
      };

    }
  }
}

#endif
