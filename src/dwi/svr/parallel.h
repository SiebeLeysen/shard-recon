#ifndef __dwi_micro_parallel_h__
#define __dwi_micro_parallel_h__

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include "progressbar.h"
#include "thread.h"

namespace MR { namespace DWI { namespace Micro {

  // run f(i) for i in [0, N) on all threads, with a progress bar updated from the calling thread
  template <class Functor>
  void parallel_for (size_t N, const std::string& message, Functor&& f)
  {
    std::atomic<size_t> next (0), done (0);
    ProgressBar progress (message, N);
    const size_t nthreads = std::max<size_t> (1, Thread::number_of_threads());
    std::vector<std::thread> threads;
    for (size_t t = 0; t < nthreads; ++t)
      threads.emplace_back ([&]() { for (size_t i; (i = next++) < N; ) { f (i); ++done; } });
    size_t shown = 0;
    while (shown < N) {
      std::this_thread::sleep_for (std::chrono::milliseconds (50));
      for (size_t d = done; shown < d; ++shown) ++progress;
    }
    for (auto& t : threads) t.join();
  }

}}}

#endif
