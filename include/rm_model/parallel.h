#ifndef RM_MODEL_PARALLEL_H
#define RM_MODEL_PARALLEL_H

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace rm_model {

void set_thread_count(std::size_t count);
std::size_t thread_count();
bool in_parallel_region();

class ScopedParallelRegion {
 public:
  ScopedParallelRegion();
  ScopedParallelRegion(const ScopedParallelRegion&) = delete;
  ScopedParallelRegion& operator=(const ScopedParallelRegion&) = delete;
  ~ScopedParallelRegion();

 private:
  bool prev_;
};

inline std::size_t worker_count(std::size_t items) {
  std::size_t threads = thread_count();
  return std::min<std::size_t>(std::max<std::size_t>(1, threads), items);
}

namespace detail {
void parallel_for_impl(std::size_t count,
                       const std::function<void(std::size_t, std::size_t)>& func);
} // namespace detail

class TaskGroup {
 public:
  explicit TaskGroup(std::size_t max_inflight = 0);
  TaskGroup(const TaskGroup&) = delete;
  TaskGroup& operator=(const TaskGroup&) = delete;
  ~TaskGroup();

  void schedule(std::function<void()> task);
  void wait();

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::size_t inflight_;
  std::size_t remaining_;
  std::size_t max_inflight_;
  bool registered_;
};

template <typename Func>
void parallel_for(std::size_t count, Func func) {
  if (count == 0) return;
  detail::parallel_for_impl(count, [&](std::size_t start, std::size_t end) {
    for (std::size_t i = start; i < end; ++i) {
      func(i);
    }
  });
}

template <typename FuncLeft, typename FuncRight>
void join(FuncLeft left, FuncRight right) {
  if (thread_count() <= 1 || in_parallel_region()) {
    left();
    right();
    return;
  }

  std::thread worker([&]() { right(); });
  left();
  worker.join();
}

} // namespace rm_model

#endif // RM_MODEL_PARALLEL_H
