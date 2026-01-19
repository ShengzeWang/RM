#ifndef RM_MODEL_CACHE_FIX_H
#define RM_MODEL_CACHE_FIX_H

#include "rm_model/training_data.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace rm_model {

std::vector<std::pair<uint64_t, std::size_t>> cache_fix(const TrainingData<uint64_t>& data,
                                                        std::size_t line_size);

} // namespace rm_model

#endif // RM_MODEL_CACHE_FIX_H
