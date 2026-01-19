#ifndef RM_MODEL_MODELS_BALANCED_RADIX_H
#define RM_MODEL_MODELS_BALANCED_RADIX_H

#include "rm_model/models/model.h"
#include "rm_model/models/utils.h"
#include "rm_model/training_data.h"

namespace rm_model {

class BalancedRadixModel : public Model {
 public:
  explicit BalancedRadixModel(const TrainingData<uint64_t>& data);
  explicit BalancedRadixModel(const TrainingData<uint32_t>& data);
  explicit BalancedRadixModel(const TrainingData<double>& data);

  uint64_t predict_to_int(const ModelInput& inp) const override;
  ModelDataType input_type() const override { return ModelDataType::Int; }
  ModelDataType output_type() const override { return ModelDataType::Int; }
  std::vector<ModelParam> params() const override;
  std::string code() const override;
 std::string function_name() const override;
 bool needs_bounds_check() const override { return false; }
 ModelRestriction restriction() const override { return ModelRestriction::MustBeTop; }

  BalancedRadixModel(uint8_t prefix, uint8_t bits, uint64_t clamp, bool high);

 private:
  std::tuple<uint8_t, uint8_t, uint64_t> params_{0, 0, 0};
  bool high_{true};
};

} // namespace rm_model

#endif // RM_MODEL_MODELS_BALANCED_RADIX_H
