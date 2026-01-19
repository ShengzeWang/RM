#ifndef RM_MODEL_MODELS_RADIX_H
#define RM_MODEL_MODELS_RADIX_H

#include "rm_model/models/model.h"
#include "rm_model/models/utils.h"
#include "rm_model/training_data.h"

namespace rm_model {

class RadixModel : public Model {
 public:
  explicit RadixModel(const TrainingData<uint64_t>& data);
  explicit RadixModel(const TrainingData<uint32_t>& data);
  explicit RadixModel(const TrainingData<double>& data);

  uint64_t predict_to_int(const ModelInput& inp) const override;
  ModelDataType input_type() const override { return ModelDataType::Int; }
  ModelDataType output_type() const override { return ModelDataType::Int; }
  std::vector<ModelParam> params() const override;
  std::string code() const override;
  std::string function_name() const override { return "radix"; }
  bool needs_bounds_check() const override { return false; }
  ModelRestriction restriction() const override { return ModelRestriction::MustBeTop; }

 private:
  std::pair<uint8_t, uint8_t> params_{0, 0};
};

class RadixTable : public Model {
 public:
  RadixTable(const TrainingData<uint64_t>& data, uint8_t bits);
  RadixTable(const TrainingData<uint32_t>& data, uint8_t bits);
  RadixTable(const TrainingData<double>& data, uint8_t bits);

  uint64_t predict_to_int(const ModelInput& inp) const override;
  ModelDataType input_type() const override { return ModelDataType::Int; }
  ModelDataType output_type() const override { return ModelDataType::Int; }
  std::vector<ModelParam> params() const override;
  std::string code() const override;
  std::string function_name() const override { return "radix_table"; }
  bool needs_bounds_check() const override { return false; }

  RadixTable(uint8_t prefix_bits, uint8_t table_bits, std::vector<uint32_t> hint_table);

 private:
  uint8_t prefix_bits_{0};
  uint8_t table_bits_{0};
  std::vector<uint32_t> hint_table_;
};

} // namespace rm_model

#endif // RM_MODEL_MODELS_RADIX_H
