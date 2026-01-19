#ifndef RM_MODEL_MODELS_HISTOGRAM_H
#define RM_MODEL_MODELS_HISTOGRAM_H

#include "rm_model/models/model.h"
#include "rm_model/models/utils.h"
#include "rm_model/models/stdlib.h"
#include "rm_model/training_data.h"

namespace rm_model {

class EquidepthHistogramModel : public Model {
 public:
  explicit EquidepthHistogramModel(const TrainingData<uint64_t>& data);
  explicit EquidepthHistogramModel(const TrainingData<uint32_t>& data);
  explicit EquidepthHistogramModel(const TrainingData<double>& data);

  uint64_t predict_to_int(const ModelInput& inp) const override;
  ModelDataType input_type() const override { return ModelDataType::Int; }
  ModelDataType output_type() const override { return ModelDataType::Int; }
  std::vector<ModelParam> params() const override;
  std::string code() const override;
  std::set<StdFunctions> standard_functions() const override;
  std::string function_name() const override { return "ed_histogram"; }
  ModelRestriction restriction() const override { return ModelRestriction::MustBeTop; }
  bool needs_bounds_check() const override { return false; }

 private:
  std::vector<uint64_t> params_;
  std::vector<uint64_t> radix_;
};

} // namespace rm_model

#endif // RM_MODEL_MODELS_HISTOGRAM_H
