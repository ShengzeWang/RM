#ifndef RM_MODEL_MODELS_CUBIC_SPLINE_H
#define RM_MODEL_MODELS_CUBIC_SPLINE_H

#include "rm_model/models/model.h"
#include "rm_model/training_data.h"

namespace rm_model {

class CubicSplineModel : public Model {
 public:
  explicit CubicSplineModel(const TrainingData<uint64_t>& data);
  explicit CubicSplineModel(const TrainingData<uint32_t>& data);
  explicit CubicSplineModel(const TrainingData<double>& data);

  double predict_to_float(const ModelInput& inp) const override;
  ModelDataType input_type() const override { return ModelDataType::Float; }
  ModelDataType output_type() const override { return ModelDataType::Float; }
  std::vector<ModelParam> params() const override;
  std::string code() const override;
  std::string function_name() const override { return "cubic"; }
  bool needs_bounds_check() const override { return false; }
  bool set_to_constant_model(uint64_t constant) override;

 private:
  std::tuple<double, double, double, double> params_{0.0, 0.0, 1.0, 0.0};
};

} // namespace rm_model

#endif // RM_MODEL_MODELS_CUBIC_SPLINE_H
