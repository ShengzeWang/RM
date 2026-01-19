#ifndef RM_MODEL_TRAINING_DATA_H
#define RM_MODEL_TRAINING_DATA_H

#include "rm_model/models/model.h"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace rm_model {

template <typename T>
struct TrainingKeyOps;

template <>
struct TrainingKeyOps<uint64_t> {
  static uint64_t minus_epsilon(uint64_t v) { return v == 0 ? 0 : v - 1; }
  static uint64_t zero_value() { return 0; }
  static uint64_t plus_epsilon(uint64_t v) {
    return v == std::numeric_limits<uint64_t>::max() ? v : v + 1;
  }
  static uint64_t max_value() { return std::numeric_limits<uint64_t>::max(); }
  static double as_float(uint64_t v) { return static_cast<double>(v); }
  static uint64_t as_uint(uint64_t v) { return v; }
  static ModelInput to_model_input(uint64_t v) { return ModelInput(v); }
};

template <>
struct TrainingKeyOps<uint32_t> {
  static uint32_t minus_epsilon(uint32_t v) { return v == 0 ? 0 : static_cast<uint32_t>(v - 1); }
  static uint32_t zero_value() { return 0; }
  static uint32_t plus_epsilon(uint32_t v) {
    return v == std::numeric_limits<uint32_t>::max() ? v : static_cast<uint32_t>(v + 1);
  }
  static uint32_t max_value() { return std::numeric_limits<uint32_t>::max(); }
  static double as_float(uint32_t v) { return static_cast<double>(v); }
  static uint64_t as_uint(uint32_t v) { return static_cast<uint64_t>(v); }
  static ModelInput to_model_input(uint32_t v) { return ModelInput(v); }
};

template <>
struct TrainingKeyOps<double> {
  static double minus_epsilon(double v) { return v - std::numeric_limits<double>::epsilon(); }
  static double zero_value() { return 0.0; }
  static double plus_epsilon(double v) { return v + std::numeric_limits<double>::epsilon(); }
  static double max_value() { return std::numeric_limits<double>::max(); }
  static double as_float(double v) { return v; }
  static uint64_t as_uint(double v) { return static_cast<uint64_t>(v); }
  static ModelInput to_model_input(double v) { return ModelInput(v); }
};

template <typename T>
class TrainingDataIteratorProvider {
 public:
  using value_type = std::pair<T, std::size_t>;

  virtual ~TrainingDataIteratorProvider() = default;
  virtual std::size_t len() const = 0;
  virtual KeyType key_type() const = 0;
  virtual bool get(std::size_t idx, T& key, std::size_t& offset) const = 0;
  virtual const value_type* raw_pairs() const { return nullptr; }
  virtual const T* raw_keys() const { return nullptr; }
};

template <typename T>
class TrainingData {
 public:
  using value_type = std::pair<T, std::size_t>;

  explicit TrainingData(std::shared_ptr<TrainingDataIteratorProvider<T>> provider)
      : provider_(std::move(provider)),
        scale_(1.0),
        scale_is_one_(true),
        length_(provider_->len()),
        raw_pairs_(provider_->raw_pairs()),
        raw_keys_(provider_->raw_keys()) {}

  static TrainingData<T> empty() {
    return TrainingData<T>(std::make_shared<VectorProvider>(std::vector<value_type>{}));
  }

  std::size_t len() const { return length_; }

  void set_scale(double scale) {
    scale_ = scale;
    scale_is_one_ = std::abs(scale_ - 1.0) <= std::numeric_limits<double>::epsilon();
  }

  value_type get(std::size_t idx) const { return apply_scale(raw_get(idx)); }

  T get_key(std::size_t idx) const {
    return get(idx).first;
  }

  class FixDupsIterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = std::pair<T, std::size_t>;
    using difference_type = std::ptrdiff_t;
    using pointer = const value_type*;
    using reference = const value_type&;

    FixDupsIterator(const TrainingData<T>* data, std::size_t index, bool end)
        : data_(data), index_(index), end_(end) {
      if (!end_) {
        advance();
      }
    }

    reference operator*() const { return current_.value(); }
    pointer operator->() const { return &current_.value(); }

    FixDupsIterator& operator++() {
      advance();
      return *this;
    }

    FixDupsIterator operator++(int) {
      FixDupsIterator tmp(*this);
      advance();
      return tmp;
    }

    bool operator==(const FixDupsIterator& other) const {
      return end_ == other.end_ && index_ == other.index_ && data_ == other.data_;
    }

    bool operator!=(const FixDupsIterator& other) const { return !(*this == other); }

   private:
    void advance() {
      if (index_ >= data_->len()) {
        end_ = true;
        current_.reset();
        return;
      }

      while (index_ < data_->len()) {
        auto raw = data_->raw_get(index_++);
        if (!last_item_) {
          last_item_ = raw;
          current_ = data_->apply_scale(raw);
          return;
        }

        if (raw.first == last_item_->first) {
          current_ = data_->apply_scale({raw.first, last_item_->second});
          return;
        }

        last_item_ = raw;
        current_ = data_->apply_scale(raw);
        return;
      }

      end_ = true;
      current_.reset();
    }

    const TrainingData<T>* data_;
    std::size_t index_;
    bool end_;
    std::optional<value_type> last_item_;
    std::optional<value_type> current_;
  };

  class DedupIterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = std::pair<T, std::size_t>;
    using difference_type = std::ptrdiff_t;
    using pointer = const value_type*;
    using reference = const value_type&;

    DedupIterator(const TrainingData<T>* data, std::size_t index, bool end)
        : data_(data), index_(index), end_(end) {
      if (!end_) {
        advance();
      }
    }

    reference operator*() const { return current_.value(); }
    pointer operator->() const { return &current_.value(); }

    DedupIterator& operator++() {
      advance();
      return *this;
    }

    DedupIterator operator++(int) {
      DedupIterator tmp(*this);
      advance();
      return tmp;
    }

    bool operator==(const DedupIterator& other) const {
      return end_ == other.end_ && index_ == other.index_ && data_ == other.data_;
    }

    bool operator!=(const DedupIterator& other) const { return !(*this == other); }

   private:
    void advance() {
      if (index_ >= data_->len()) {
        end_ = true;
        current_.reset();
        return;
      }

      while (index_ < data_->len()) {
        auto raw = data_->raw_get(index_++);
        if (!last_item_) {
          last_item_ = raw;
          current_ = data_->apply_scale(raw);
          return;
        }

        if (raw.first == last_item_->first) {
          continue;
        }

        last_item_ = raw;
        current_ = data_->apply_scale(raw);
        return;
      }

      end_ = true;
      current_.reset();
    }

    const TrainingData<T>* data_;
    std::size_t index_;
    bool end_;
    std::optional<value_type> last_item_;
    std::optional<value_type> current_;
  };

  class FixDupsRange {
   public:
    explicit FixDupsRange(const TrainingData<T>* data) : data_(data) {}
    FixDupsIterator begin() const { return FixDupsIterator(data_, 0, data_->len() == 0); }
    FixDupsIterator end() const { return FixDupsIterator(data_, data_->len(), true); }

   private:
    const TrainingData<T>* data_;
  };

  class DedupRange {
   public:
    explicit DedupRange(const TrainingData<T>* data) : data_(data) {}
    DedupIterator begin() const { return DedupIterator(data_, 0, data_->len() == 0); }
    DedupIterator end() const { return DedupIterator(data_, data_->len(), true); }

   private:
    const TrainingData<T>* data_;
  };

  FixDupsRange iter() const { return FixDupsRange(this); }
  DedupRange iter_unique() const { return DedupRange(this); }

  class ModelInputIterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = std::pair<ModelInput, std::size_t>;
    using difference_type = std::ptrdiff_t;
    using pointer = const value_type*;
    using reference = const value_type&;

    ModelInputIterator(FixDupsIterator iter, FixDupsIterator end)
        : iter_(std::move(iter)), end_(std::move(end)) {
      if (iter_ != end_) {
        update();
      }
    }

    reference operator*() const { return current_; }
    pointer operator->() const { return &current_; }

    ModelInputIterator& operator++() {
      ++iter_;
      if (iter_ != end_) {
        update();
      }
      return *this;
    }

    bool operator==(const ModelInputIterator& other) const { return iter_ == other.iter_; }
    bool operator!=(const ModelInputIterator& other) const { return !(*this == other); }

   private:
    void update() {
      auto [key, offset] = *iter_;
      current_ = {TrainingKeyOps<T>::to_model_input(key), offset};
    }

    FixDupsIterator iter_;
    FixDupsIterator end_;
    value_type current_{};
  };

  class ModelInputRange {
   public:
    explicit ModelInputRange(const TrainingData<T>* data) : data_(data) {}
    ModelInputIterator begin() const {
      return ModelInputIterator(data_->iter().begin(), data_->iter().end());
    }
    ModelInputIterator end() const {
      return ModelInputIterator(data_->iter().end(), data_->iter().end());
    }

   private:
    const TrainingData<T>* data_;
  };

  ModelInputRange iter_model_input() const { return ModelInputRange(this); }

  template <typename Comparator>
  std::size_t lower_bound_by(Comparator comp) const {
    std::size_t size = len();
    if (size == 0) return 0;
    std::size_t base = 0;
    while (size > 1) {
      std::size_t half = size / 2;
      std::size_t mid = base + half;
      int cmp = comp(get(mid));
      base = (cmp < 0) ? mid : base;
      size -= half;
    }
    int cmp = comp(get(base));
    return base + static_cast<std::size_t>(cmp < 0);
  }

  TrainingData<T> soft_copy() const {
    TrainingData<T> copy(provider_);
    copy.scale_ = scale_;
    copy.scale_is_one_ = scale_is_one_;
    copy.length_ = length_;
    return copy;
  }

  KeyType key_type() const { return provider_->key_type(); }

  class VectorProvider : public TrainingDataIteratorProvider<T> {
   public:
    explicit VectorProvider(std::vector<value_type> data) : data_(std::move(data)) {}

    std::size_t len() const override { return data_.size(); }
    KeyType key_type() const override {
      if constexpr (std::is_same_v<T, uint64_t>) return KeyType::U64;
      if constexpr (std::is_same_v<T, uint32_t>) return KeyType::U32;
      return KeyType::F64;
    }
    bool get(std::size_t idx, T& key, std::size_t& offset) const override {
      if (idx >= data_.size()) return false;
      key = data_[idx].first;
      offset = data_[idx].second;
      return true;
    }
    const value_type* raw_pairs() const override {
      return data_.empty() ? nullptr : data_.data();
    }

   private:
    std::vector<value_type> data_;
  };

 private:
  value_type raw_get(std::size_t idx) const {
    if (idx >= length_) {
      throw std::out_of_range("Index out of range");
    }
    if (raw_pairs_) {
      return raw_pairs_[idx];
    }
    if (raw_keys_) {
      return {raw_keys_[idx], idx};
    }
    T key{};
    std::size_t offset = 0;
    if (!provider_->get(idx, key, offset)) {
      throw std::out_of_range("Index out of range");
    }
    return {key, offset};
  }

  value_type apply_scale(value_type item) const {
    if (scale_is_one_) {
      return item;
    }
    item.second = static_cast<std::size_t>(static_cast<double>(item.second) * scale_);
    return item;
  }

  std::shared_ptr<TrainingDataIteratorProvider<T>> provider_;
  double scale_;
  bool scale_is_one_;
  std::size_t length_;
  const value_type* raw_pairs_;
  const T* raw_keys_;
};

} // namespace rm_model

#endif // RM_MODEL_TRAINING_DATA_H
