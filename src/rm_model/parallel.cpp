#include "rm_model/parallel.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace rm_model {

namespace {
std::atomic<std::size_t> g_thread_count{4};
std::atomic<std::size_t> g_active_tasks{0};
thread_local bool g_in_parallel = false;

class ThreadPool {
 public:
  explicit ThreadPool(std::size_t threads) : stop_(false) {
    threads = std::max<std::size_t>(1, threads);
    workers_.reserve(threads);
    for (std::size_t i = 0; i < threads; ++i) {
      workers_.emplace_back([this]() { worker_loop(); });
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& worker : workers_) {
      worker.join();
    }
  }

  void enqueue(std::function<void()> task) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (stop_) {
        throw std::runtime_error("enqueue on stopped ThreadPool");
      }
      tasks_.push_back(std::move(task));
    }
    cv_.notify_one();
  }

  std::size_t size() const { return workers_.size(); }

 private:
  void worker_loop() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&]() { return stop_ || !tasks_.empty(); });
        if (stop_ && tasks_.empty()) return;
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      task();
    }
  }

  std::vector<std::thread> workers_;
  std::deque<std::function<void()>> tasks_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_;
};

std::mutex g_pool_mutex;
std::unique_ptr<ThreadPool> g_pool;

ThreadPool& pool() {
  std::lock_guard<std::mutex> lock(g_pool_mutex);
  if (!g_pool) {
    g_pool = std::make_unique<ThreadPool>(g_thread_count.load());
  }
  return *g_pool;
}

void reset_pool(std::size_t threads) {
  std::lock_guard<std::mutex> lock(g_pool_mutex);
  g_pool = std::make_unique<ThreadPool>(threads);
}
} // namespace

void set_thread_count(std::size_t count) {
  std::size_t new_count = std::max<std::size_t>(1, count);
  std::size_t old_count = g_thread_count.exchange(new_count);
  if (old_count == new_count) {
    return;
  }
  if (g_active_tasks.load() == 0) {
    reset_pool(new_count);
  }
}

std::size_t thread_count() {
  return g_thread_count.load();
}

bool in_parallel_region() {
  return g_in_parallel;
}

ScopedParallelRegion::ScopedParallelRegion() : prev_(g_in_parallel) {
  g_in_parallel = true;
}

ScopedParallelRegion::~ScopedParallelRegion() {
  g_in_parallel = prev_;
}

TaskGroup::TaskGroup(std::size_t max_inflight)
    : inflight_(0),
      remaining_(0),
      max_inflight_(max_inflight == 0 ? std::max<std::size_t>(1, thread_count() * 2) : max_inflight),
      registered_(true) {
  g_active_tasks.fetch_add(1);
}

TaskGroup::~TaskGroup() {
  wait();
  if (registered_) {
    g_active_tasks.fetch_sub(1);
  }
}

void TaskGroup::schedule(std::function<void()> task) {
  if (!task) return;

  {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&]() { return inflight_ < max_inflight_; });
    ++inflight_;
    ++remaining_;
  }

  pool().enqueue([this, task = std::move(task)]() mutable {
    task();
    {
      std::lock_guard<std::mutex> lock(mu_);
      --inflight_;
      --remaining_;
    }
    cv_.notify_all();
  });
}

void TaskGroup::wait() {
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait(lock, [&]() { return remaining_ == 0; });
}

namespace detail {

void parallel_for_impl(std::size_t count,
                       const std::function<void(std::size_t, std::size_t)>& func) {
  if (count == 0) return;
  struct TaskScope {
    TaskScope() { g_active_tasks.fetch_add(1); }
    ~TaskScope() { g_active_tasks.fetch_sub(1); }
  } scope;

  std::size_t threads = worker_count(count);
  if (threads <= 1) {
    ScopedParallelRegion guard;
    func(0, count);
    return;
  }

  std::size_t grain = std::max<std::size_t>(1, count / (threads * 4));
  std::size_t num_tasks = (count + grain - 1) / grain;
  if (num_tasks <= 1) {
    func(0, count);
    return;
  }

  std::atomic<std::size_t> remaining{num_tasks};
  std::mutex done_mutex;
  std::condition_variable done_cv;

  auto& tp = pool();
  for (std::size_t task_idx = 0; task_idx < num_tasks; ++task_idx) {
    std::size_t start = task_idx * grain;
    std::size_t end = std::min<std::size_t>(count, start + grain);
    tp.enqueue([&, start, end]() {
      ScopedParallelRegion guard;
      func(start, end);
      if (remaining.fetch_sub(1) == 1) {
        done_cv.notify_one();
      }
    });
  }

  std::unique_lock<std::mutex> lock(done_mutex);
  done_cv.wait(lock, [&]() { return remaining.load() == 0; });
}

} // namespace detail

} // namespace rm_model
