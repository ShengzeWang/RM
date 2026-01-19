#ifndef RM_MODEL_MODELS_NORMAL_H
#define RM_MODEL_MODELS_NORMAL_H

#include "rm_model/models/model.h"
#include "rm_model/models/stdlib.h"
#include "rm_model/training_data.h"

namespace rm_model {

class NormalModel : public Model {
 public:
  explicit NormalModel(const TrainingData<uint64_t>& data);
  explicit NormalModel(const TrainingData<uint32_t>& data);
  explicit NormalModel(const TrainingData<double>& data);

  double predict_to_float(const ModelInput& inp) const override;
  ModelDataType input_type() const override { return ModelDataType::Float; }
  ModelDataType output_type() const override { return ModelDataType::Float; }
  std::vector<ModelParam> params() const override;
  std::string code() const override;
  std::string function_name() const override { return "ncdf"; }
  std::set<StdFunctions> standard_functions() const override;

 private:
  std::tuple<double, double, double> params_{0.0, 0.0, 0.0};
};

class LogNormalModel : public Model {
 public:
  explicit LogNormalModel(const TrainingData<uint64_t>& data);
  explicit LogNormalModel(const TrainingData<uint32_t>& data);
  explicit LogNormalModel(const TrainingData<double>& data);

  double predict_to_float(const ModelInput& inp) const override;
  ModelDataType input_type() const override { return ModelDataType::Float; }
  ModelDataType output_type() const override { return ModelDataType::Float; }
  std::vector<ModelParam> params() const override;
  std::string code() const override;
  std::string function_name() const override { return "lncdf"; }
  std::set<StdFunctions> standard_functions() const override;

 private:
  std::tuple<double, double, double> params_{0.0, 0.0, 0.0};
};

} // namespace rm_model

#endif // RM_MODEL_MODELS_NORMAL_H
