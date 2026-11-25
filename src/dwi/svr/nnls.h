/* Non-negative least squares from the normal equations:
 *     minimise 0.5 x^T AtA x - Atb^T x   subject to x >= 0
 * Active-set algorithm of Lawson & Hanson in the "fast NNLS" form of Bro & De Jong (1997), which only
 * needs AtA and Atb, so that problems sharing a design (or built from precomputed blocks) are cheap.
 */

#ifndef __dwi_micro_nnls_h__
#define __dwi_micro_nnls_h__

#include <vector>
#include <Eigen/Dense>

namespace MR { namespace DWI { namespace Micro {

  inline Eigen::VectorXd nnls (const Eigen::MatrixXd& AtA, const Eigen::VectorXd& Atb, int max_iterations = 0)
  {
    const int K = Atb.size();
    if (max_iterations <= 0) max_iterations = 3*K;
    Eigen::VectorXd x = Eigen::VectorXd::Zero (K);
    std::vector<bool> passive (K, false);
    const double tol = 1e-12 * std::max (1.0, Atb.cwiseAbs().maxCoeff()) * std::max (1.0, AtA.diagonal().maxCoeff());
    Eigen::VectorXd w = Atb;

    auto solve_passive = [&](Eigen::VectorXd& s) {
      std::vector<int> P;
      for (int i = 0; i < K; ++i) if (passive[i]) P.push_back (i);
      Eigen::MatrixXd M (P.size(), P.size());
      Eigen::VectorXd r (P.size());
      for (size_t a = 0; a < P.size(); ++a) {
        r[a] = Atb[P[a]];
        for (size_t b = 0; b < P.size(); ++b) M(a,b) = AtA(P[a], P[b]);
      }
      const Eigen::VectorXd sp = M.ldlt().solve (r);
      s.setZero (K);
      for (size_t a = 0; a < P.size(); ++a) s[P[a]] = sp[a];
    };

    for (int outer = 0; outer < max_iterations; ++outer) {
      int t = -1;
      double best = tol;
      for (int i = 0; i < K; ++i)
        if (!passive[i] && w[i] > best) { best = w[i]; t = i; }
      if (t < 0) break;
      passive[t] = true;
      Eigen::VectorXd s;
      for (int inner = 0; inner < K; ++inner) {
        solve_passive (s);
        double alpha = 1.0;
        bool feasible = true;
        for (int i = 0; i < K; ++i)
          if (passive[i] && s[i] <= 0.0) {
            feasible = false;
            const double d = x[i] - s[i];
            if (d > 0.0) alpha = std::min (alpha, x[i] / d);
          }
        if (feasible) { x = s; break; }
        x += alpha * (s - x);
        for (int i = 0; i < K; ++i)
          if (passive[i] && x[i] <= 1e-15 * std::max (1.0, x.cwiseAbs().maxCoeff())) { passive[i] = false; x[i] = 0.0; }
      }
      w = Atb - AtA * x;
    }
    return x;
  }

}}}

#endif
