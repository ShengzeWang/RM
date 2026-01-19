#ifndef RM_MODEL_PROGRESS_H
#define RM_MODEL_PROGRESS_H

#include <atomic>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace rm_model {

class ProgressBar {
 public:
  explicit ProgressBar(std::size_t total)
      : total_(total),
        current_(0),
        last_print_ms_(now_ms()),
        is_tty_(stderr_is_tty()) {}

  void set_message(const std::string& msg) {
    std::lock_guard<std::mutex> lock(mu_);
    message_ = msg;
  }

  void inc(std::size_t delta = 1) {
    inc_impl(delta, false);
  }

  void update_message(const std::string& msg) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      message_ = msg;
    }
    maybe_print(true);
  }

  void inc_with_message(const std::string& msg, std::size_t delta = 1) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      message_ = msg;
    }
    inc_impl(delta, true);
  }

  void finish() {
    std::lock_guard<std::mutex> lock(mu_);
    print_line_locked(true);
  }

 private:
  static bool stderr_is_tty() {
#if defined(_WIN32)
    return _isatty(_fileno(stderr)) != 0;
#else
    return isatty(fileno(stderr)) != 0;
#endif
  }

  static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  void inc_impl(std::size_t delta, bool force_print) {
    std::size_t current = current_.fetch_add(delta) + delta;
    if (force_print || current == delta || current >= total_) {
      maybe_print(true);
      return;
    }
    maybe_print(false);
  }

  void maybe_print(bool force) {
    int64_t now = now_ms();
    int64_t last = last_print_ms_.load(std::memory_order_relaxed);
    if (!force) {
      if (now - last < 250) return;
      if (!last_print_ms_.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
        return;
      }
    } else {
      last_print_ms_.store(now, std::memory_order_relaxed);
    }

    std::lock_guard<std::mutex> lock(mu_);
    print_line_locked(false);
  }

  void print_line_locked(bool final) const {
    if (is_tty_) {
      std::cerr << "\r" << format_line();
      if (final) {
        std::cerr << "\n";
      }
      std::cerr << std::flush;
      return;
    }
    std::cerr << format_line() << "\n";
  }

  std::string format_line() const {
    std::ostringstream oss;
    std::size_t current = current_.load();
    std::size_t total = total_;
    double frac = total == 0 ? 1.0 : static_cast<double>(current) / static_cast<double>(total);
    if (frac < 0.0) frac = 0.0;
    if (frac > 1.0) frac = 1.0;
    int pct = static_cast<int>(frac * 100.0 + 0.5);
    const std::size_t width = 28;
    std::size_t filled = static_cast<std::size_t>(frac * static_cast<double>(width));
    oss << current << " / " << total << " (" << pct << "%) [";
    for (std::size_t i = 0; i < width; ++i) {
      oss << (i < filled ? '#' : '-');
    }
    oss << "]";
    if (!message_.empty()) {
      oss << " (" << message_ << ")";
    }
    return oss.str();
  }

  std::size_t total_;
  std::atomic<std::size_t> current_;
  std::atomic<int64_t> last_print_ms_;
  bool is_tty_;
  std::string message_;
  std::mutex mu_;
};

} // namespace rm_model

#endif // RM_MODEL_PROGRESS_H
