#ifndef RM_MODEL_MODELS_LINEAR_H
#define RM_MODEL_MODELS_LINEAR_H

#include "rm_model/models/model.h"
#include "rm_model/models/stdlib.h"
#include "rm_model/training_data.h"

namespace rm_model {

class LinearModel : public Model {
 public:
  explicit LinearModel(const TrainingData<uint64_t>& data);
  explicit LinearModel(const TrainingData<uint32_t>& data);
  explicit LinearModel(const TrainingData<double>& data);

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

class LogLinearModel : public Model {
 public:
  explicit LogLinearModel(const TrainingData<uint64_t>& data);
  explicit LogLinearModel(const TrainingData<uint32_t>& data);
  explicit LogLinearModel(const TrainingData<double>& data);

  double predict_to_float(const ModelInput& inp) const override;
  ModelDataType input_type() const override { return ModelDataType::Float; }
  ModelDataType output_type() const override { return ModelDataType::Float; }
  std::vector<ModelParam> params() const override;
  std::string code() const override;
  std::string function_name() const override { return "loglinear"; }
  std::set<StdFunctions> standard_functions() const override;

 private:
  std::pair<double, double> params_{0.0, 0.0};
};

class RobustLinearModel : public Model {
 public:
  explicit RobustLinearModel(const TrainingData<uint64_t>& data);
  explicit RobustLinearModel(const TrainingData<uint32_t>& data);
  explicit RobustLinearModel(const TrainingData<double>& data);

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

#endif // RM_MODEL_MODELS_LINEAR_H
