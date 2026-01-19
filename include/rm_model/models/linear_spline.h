#ifndef RM_MODEL_MODELS_LINEAR_SPLINE_H
#define RM_MODEL_MODELS_LINEAR_SPLINE_H

#include "rm_model/models/model.h"
#include "rm_model/training_data.h"

namespace rm_model {

class LinearSplineModel : public Model {
 public:
  explicit LinearSplineModel(const TrainingData<uint64_t>& data);
  explicit LinearSplineModel(const TrainingData<uint32_t>& data);
  explicit LinearSplineModel(const TrainingData<double>& data);

  double predict_to_float(const ModelInput& inp) const override;
  ModelDataType input_type() const override { return ModelDataType::Float; }
  ModelDataType output_type() const override { return ModelDataType::Float; }
  std::vector<ModelParam> params() const override;
  std::string code() const override;
  std::string function_name() const override { return "linear"; }
  bool set_to_constant_model(uint64_t constant) override;

 private:
  std::pair<double, double> params_{0.0, 0.0};
};

} // namespace rm_model

#endif // RM_MODEL_MODELS_LINEAR_SPLINE_H
