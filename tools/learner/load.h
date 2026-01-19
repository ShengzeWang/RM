#ifndef RM_MODEL_LOAD_H
#define RM_MODEL_LOAD_H

#include "rm_model/training_data.h"

#include <optional>
#include <string>
#include <utility>
#include <variant>

enum class DataType {
  UINT64,
  UINT32,
  FLOAT64,
};

class MappedDataset {
 public:
  using U64Data = rm_model::TrainingData<uint64_t>;
  using U32Data = rm_model::TrainingData<uint32_t>;
  using F64Data = rm_model::TrainingData<double>;

  MappedDataset(U64Data data) : data_(std::move(data)) {}
  MappedDataset(U32Data data) : data_(std::move(data)) {}
  MappedDataset(F64Data data) : data_(std::move(data)) {}

  MappedDataset soft_copy() const;
  std::optional<U64Data> into_u64() const;

  template <typename Func>
  auto visit(Func&& func) const {
    return std::visit(std::forward<Func>(func), data_);
  }

  template <typename Func>
  auto visit(Func&& func) {
    return std::visit(std::forward<Func>(func), data_);
  }

 private:
  std::variant<U64Data, U32Data, F64Data> data_;
};

std::pair<std::size_t, MappedDataset> load_data(const std::string& filepath, DataType dt);

#endif // RM_MODEL_LOAD_H
