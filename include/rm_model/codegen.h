#ifndef RM_MODEL_CODEGEN_H
#define RM_MODEL_CODEGEN_H

#include "rm_model/models/model.h"
#include "rm_model/train.h"

#include <cstdint>
#include <string>

namespace rm_model {

uint64_t model_size_bytes(const TrainedModel& model);

void emit_model(const std::string& namespace_name,
                TrainedModel trained_model,
                const std::string& output_dir,
                const std::string& data_dir,
                KeyType key_type,
                bool include_errors);

} // namespace rm_model

#endif // RM_MODEL_CODEGEN_H
