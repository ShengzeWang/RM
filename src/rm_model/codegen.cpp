#include "rm_model/codegen.h"

#include "rm_model/json.h"
#include "rm_model/logging.h"
#include "rm_model/models/stdlib.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace rm_model {

namespace {

std::string constant_name(std::size_t layer, std::size_t idx) {
  return "L" + std::to_string(layer) + "_PARAMETER" + std::to_string(idx);
}

std::string array_name(std::size_t layer) {
  return "L" + std::to_string(layer) + "_PARAMETERS";
}

class LayerParams {
 public:
  enum class Kind { Constant, Array, MixedArray };

  static LayerParams create(std::size_t idx,
                            bool array_access,
                            std::size_t params_per_model,
                            std::vector<ModelParam> params) {
    if (params.empty()) {
      throw std::runtime_error("Layer parameters cannot be empty");
    }

    bool mixed = false;
    for (std::size_t i = 1; i < params.size(); ++i) {
      if (!params[0].is_same_type(params[i])) {
        mixed = true;
        break;
      }
    }

    if (mixed) {
      return LayerParams(Kind::MixedArray, idx, params_per_model, std::move(params));
    }

    std::size_t param_size_bytes = 0;
    for (const auto& param : params) {
      param_size_bytes += param.size();
    }

    if (array_access || param_size_bytes > 4096) {
      return LayerParams(Kind::Array, idx, params_per_model, std::move(params));
    }

    return LayerParams(Kind::Constant, idx, params_per_model, std::move(params));
  }

  Kind kind() const { return kind_; }

  void to_code(std::ostream& out) const {
    if (kind_ == Kind::Constant) {
      for (std::size_t i = 0; i < params_.size(); ++i) {
        out << "const " << params_[i].c_type() << " " << constant_name(index_, i)
            << params_[i].c_type_mod() << " = " << params_[i].c_val() << ";\n";
      }
      return;
    }

    if (kind_ == Kind::Array) {
      out << "const " << params_[0].c_type() << " " << array_name(index_) << "[] = {";
      for (std::size_t i = 0; i < params_.size(); ++i) {
        if (i > 0) out << ",";
        out << params_[i].c_val();
      }
      out << "};\n";
      return;
    }

    throw std::runtime_error("Cannot hardcode mixed array");
  }

  bool requires_malloc() const {
    if (kind_ == Kind::Array) {
      std::size_t array_size = 0;
      for (const auto& param : params_) {
        array_size += param.size();
      }
      return array_size >= 4 * 1024;
    }
    if (kind_ == Kind::MixedArray) return true;
    return false;
  }

  const char* pointer_type() const {
    if (!requires_malloc()) throw std::runtime_error("No pointer type for non-malloc params");
    if (kind_ == Kind::Array) return params_[0].c_type();
    if (kind_ == Kind::MixedArray) return "char";
    throw std::runtime_error("No pointer type for constant params");
  }

  void to_decl(std::ostream& out) const {
    if (kind_ == Kind::Constant) {
      throw std::runtime_error("Cannot forward declare constants");
    }

    if (kind_ == Kind::Array) {
      if (!requires_malloc()) {
        std::size_t num_items = 0;
        for (const auto& param : params_) {
          num_items += param.len();
        }
        out << params_[0].c_type() << " " << array_name(index_) << "[" << num_items << "];\n";
      } else {
        out << params_[0].c_type() << "* " << array_name(index_) << ";\n";
      }
      return;
    }

    if (kind_ == Kind::MixedArray) {
      out << "char* " << array_name(index_) << ";\n";
      return;
    }
  }

  void write_to(std::ostream& out) const {
    if (kind_ == Kind::Constant) {
      throw std::runtime_error("Cannot write constant parameters to binary file");
    }

    for (std::size_t i = 0; i < params_.size(); ++i) {
      if (kind_ == Kind::Array && i > 0 && !params_[0].is_same_type(params_[i])) {
        throw std::runtime_error("Mixed types in Array params");
      }
      params_[i].write_to(out);
    }
  }

  std::size_t size() const {
    std::size_t total = 0;
    for (const auto& param : params_) {
      total += param.size();
    }
    return total;
  }

  const std::vector<ModelParam>& params() const { return params_; }

  std::size_t index() const { return index_; }

  std::size_t params_per_model() const { return params_per_model_; }

  void access_by_const(std::ostream& out, std::size_t parameter_index) const {
    if (kind_ == Kind::Constant) {
      out << constant_name(index_, parameter_index);
      return;
    }
    access_by_ref(out, "0", parameter_index);
  }

  void access_by_ref(std::ostream& out, const std::string& model_index, std::size_t parameter_index) const {
    if (!params_.empty() && params_[0].is_array()) {
      if (params_.size() != 1) {
        throw std::runtime_error("Layer params with array had more than one member");
      }
      out << array_name(index_);
      return;
    }

    if (kind_ == Kind::Constant) {
      throw std::runtime_error("Cannot access constant parameters by reference");
    }

    if (kind_ == Kind::Array) {
      std::ostringstream expr;
      expr << params_per_model_ << "*" << model_index << " + " << parameter_index;
      out << array_name(index_) << "[" << expr.str() << "]";
      return;
    }

    if (kind_ == Kind::MixedArray) {
      std::size_t bytes_per_model = 0;
      for (std::size_t i = 0; i < params_per_model_; ++i) {
        bytes_per_model += params_[i].size();
      }

      std::size_t offset = 0;
      for (std::size_t i = 0; i < parameter_index; ++i) {
        offset += params_[i].size();
      }

      std::string c_type = params_[parameter_index].c_type();
      std::ostringstream ptr_expr;
      ptr_expr << array_name(index_) << " + (" << model_index << " * " << bytes_per_model
               << ") + " << offset;
      if (params_[parameter_index].is_array()) {
        out << "(" << c_type << "*) (" << ptr_expr.str() << ")";
      } else {
        out << "*(" << c_type << "*) (" << ptr_expr.str() << ")";
      }
      return;
    }
  }

  LayerParams with_zipped_errors(const std::vector<uint64_t>& lle) const {
    std::vector<ModelParam> combined;
    combined.reserve(params_.size() + lle.size());

    for (std::size_t i = 0; i < lle.size(); ++i) {
      std::size_t start = i * params_per_model_;
      for (std::size_t j = 0; j < params_per_model_; ++j) {
        combined.push_back(params_[start + j]);
      }
      combined.emplace_back(lle[i]);
    }

    bool is_constant = (kind_ == Kind::Constant);
    return LayerParams::create(index_, is_constant, params_per_model_ + 1, std::move(combined));
  }

 private:
  LayerParams(Kind kind, std::size_t idx, std::size_t ppm, std::vector<ModelParam> params)
      : kind_(kind), index_(idx), params_per_model_(ppm), params_(std::move(params)) {}

  Kind kind_;
  std::size_t index_;
  std::size_t params_per_model_;
  std::vector<ModelParam> params_;
};

LayerParams params_for_layer(std::size_t layer_idx,
                             const std::vector<std::unique_ptr<Model>>& models) {
  std::size_t params_per_model = models[0]->params().size();
  std::vector<ModelParam> params;
  params.reserve(models.size() * params_per_model);
  for (const auto& model : models) {
    auto model_params = model->params();
    params.insert(params.end(), model_params.begin(), model_params.end());
  }
  return LayerParams::create(layer_idx, models.size() > 1, params_per_model, std::move(params));
}

std::string model_index_from_output(ModelDataType from, std::size_t bound, bool needs_check) {
  if (from == ModelDataType::Float) {
    if (needs_check) {
      return "FCLAMP(fpred, " + std::to_string(bound) + ".0 - 1.0)";
    }
    return "(uint64_t) fpred";
  }

  if (from == ModelDataType::Int) {
    if (needs_check) {
      return "(ipred > " + std::to_string(bound) + " - 1 ? " + std::to_string(bound) + " - 1 : ipred)";
    }
    return "ipred";
  }

  if (from == ModelDataType::Int128) {
    if (needs_check) {
      return "(i128pred > " + std::to_string(bound) + " - 1 ? " + std::to_string(bound) + " - 1 : i128pred)";
    }
    return "i128pred";
  }

  return "ipred";
}

std::string key_type_name(KeyType key_type) {
  switch (key_type) {
    case KeyType::U64:
      return "u64";
    case KeyType::U32:
      return "u32";
    case KeyType::F64:
      return "f64";
    case KeyType::U128:
      return "u128";
  }
  return "unknown";
}

std::string shared_library_name(const std::string& ns) {
#if defined(_WIN32)
  return ns + ".dll";
#elif defined(__APPLE__)
  return "lib" + ns + ".dylib";
#else
  return "lib" + ns + ".so";
#endif
}

std::string maybe_relative(const std::filesystem::path& base,
                           const std::filesystem::path& target) {
  if (!base.is_absolute() || !target.is_absolute()) {
    return target.string();
  }
  std::error_code ec;
  auto rel = std::filesystem::relative(target, base, ec);
  if (!ec && !rel.empty() && rel.native()[0] != '.') {
    return rel.string();
  }
  return target.string();
}

std::string manifest_path(const std::filesystem::path& base,
                          const std::filesystem::path& target) {
  std::filesystem::path abs_base = std::filesystem::absolute(base);
  std::filesystem::path abs_target = target.is_absolute()
      ? target
      : (abs_base / target);
  return maybe_relative(abs_base, abs_target);
}

void write_manifest(const std::filesystem::path& output_root,
                    const std::filesystem::path& src_dir,
                    const std::filesystem::path& include_dir,
                    const std::filesystem::path& data_dir,
                    const std::string& namespace_name,
                    const TrainedModel& model,
                    KeyType key_type,
                    bool include_errors) {
  std::filesystem::path abs_root = std::filesystem::absolute(output_root);
  std::filesystem::path abs_src_dir = std::filesystem::absolute(src_dir);
  std::filesystem::path abs_include_dir = std::filesystem::absolute(include_dir);
  std::filesystem::path abs_data_dir = data_dir.is_absolute()
      ? data_dir
      : std::filesystem::absolute(data_dir);

  json::Value::Object obj;
  obj.emplace_back("schema_version", json::Value(static_cast<uint64_t>(1)));
  obj.emplace_back("name", json::Value(namespace_name));
  obj.emplace_back("model_spec", json::Value(model.model_spec));
  obj.emplace_back("branching_factor", json::Value(model.branching_factor));
  obj.emplace_back("key_type", json::Value(key_type_name(key_type)));
  obj.emplace_back("include_errors", json::Value(include_errors));
  obj.emplace_back("model_size_bytes", json::Value(model_size_bytes(model)));
  obj.emplace_back("build_time_ns", json::Value(model.build_time_ns));

  json::Value::Object paths;
  paths.emplace_back("src",
                     json::Value(manifest_path(abs_root,
                                               abs_src_dir / (namespace_name + ".cpp"))));
  json::Value::Array includes;
  includes.emplace_back(manifest_path(abs_root,
                                      abs_include_dir / (namespace_name + ".h")));
  includes.emplace_back(manifest_path(abs_root,
                                      abs_include_dir / (namespace_name + "_data.h")));
  paths.emplace_back("include", json::Value(std::move(includes)));
  paths.emplace_back("include_dir", json::Value(manifest_path(abs_root, abs_include_dir)));
  paths.emplace_back("data_dir", json::Value(manifest_path(abs_root, abs_data_dir)));
  paths.emplace_back("lib_dir", json::Value(manifest_path(abs_root, abs_root / "lib")));
  paths.emplace_back("library", json::Value("lib/" + shared_library_name(namespace_name)));
  obj.emplace_back("paths", json::Value(std::move(paths)));

  if (model.cache_fix.has_value()) {
    json::Value::Object cache;
    cache.emplace_back("line_size", json::Value(static_cast<uint64_t>(model.cache_fix->first)));
    obj.emplace_back("cache_fix", json::Value(std::move(cache)));
  }

  std::ofstream manifest(output_root / "model.json");
  if (!manifest) {
    throw std::runtime_error("Could not write model manifest");
  }
  json::Value json_root(std::move(obj));
  json::write(manifest, json_root);
}

std::string format_bytes(uint64_t bytes) {
  static const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
  double count = static_cast<double>(bytes);
  int idx = 0;
  while (count >= 1024.0 && idx < 4) {
    count /= 1024.0;
    idx += 1;
  }
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << count << " " << suffixes[idx];
  return oss.str();
}

void generate_cache_fix_code(std::ostream& out,
                             const TrainedModel& model,
                             const std::string& array_name_value) {
  std::size_t num_splines = model.cache_fix->second.size();
  std::size_t line_size = model.cache_fix->first;
  std::size_t total_keys = model.num_data_rows;

  out << R"(
#if defined(_MSC_VER)
#pragma pack(push, 1)
struct SplinePoint {
  uint64_t key;
  uint64_t value;
};
#pragma pack(pop)
#else
struct __attribute__((packed)) SplinePoint {
  uint64_t key;
  uint64_t value;
};
#endif

uint64_t lookup(uint64_t key, size_t* err) {
  const uint64_t num_spline_pts = )" << num_splines << ";\n"
      << "  const uint64_t total_keys = " << total_keys << ";\n"
      << "  size_t error_on_spline_search;\n\n"
      << "  struct SplinePoint* begin = (struct SplinePoint*) " << array_name_value << ";\n\n"
      << "  *err = " << line_size << ";\n"
      << "  uint64_t start = _rm_model_lookup_pre_cachefix(key, &error_on_spline_search);\n\n"
      << "  size_t upper = (start + error_on_spline_search > num_spline_pts\n"
      << "                  ? num_spline_pts : start + error_on_spline_search);\n"
      << "  size_t lower = (error_on_spline_search > start\n"
      << "                  ? 0 : start - error_on_spline_search);\n\n"
      << "  struct SplinePoint* res = std::lower_bound(begin + lower,\n"
      << "                                             begin + upper,\n"
      << "                                             key,\n"
      << "                                             [](const auto& lhs, const auto rhs) { return lhs.key < rhs; });\n\n"
      << "  if (res == begin + num_spline_pts)\n"
      << "    return total_keys - 1;\n"
      << "  if (res == begin) {\n"
      << "    auto pt = *res;\n"
      << "    return (((uint64_t) pt.value) / " << line_size << ") * " << line_size << ";\n"
      << "  }\n\n"
      << "  auto pt1 = *(res - 1);\n"
      << "  auto pt2 = *res;\n"
      << "  if (pt2.key == pt1.key) {\n"
      << "    return (((uint64_t) pt2.value) / " << line_size << ") * " << line_size << ";\n"
      << "  }\n\n"
      << "  auto v0 = (double)pt1.value;\n"
      << "  auto v1 = (double)pt2.value;\n"
      << "  auto t = ((double)(key - pt1.key)) / (double)(pt2.key - pt1.key);\n"
      << "  return (((uint64_t) std::fma(1.0 - t, v0, t * v1)) / " << line_size << ") * "
      << line_size << ";\n}"
      << "\n";
}

void generate_code(std::ostream& code_output,
                   std::ostream& data_output,
                   std::ostream& header_output,
                   const std::string& namespace_name,
                   const TrainedModel& model,
                   const std::string& data_dir,
                   KeyType key_type) {
  std::vector<LayerParams> layer_params;
  layer_params.reserve(model.model_layers.size());
  for (std::size_t i = 0; i < model.model_layers.size(); ++i) {
    layer_params.push_back(params_for_layer(i, model.model_layers[i]));
  }

  bool report_last_layer_errors = !model.last_layer_max_l1s.empty();
  std::ostringstream report_lle;

  if (report_last_layer_errors) {
    const auto& lle = model.last_layer_max_l1s;
    if (lle.size() > 1) {
      auto old_last = layer_params.back();
      layer_params.pop_back();
      auto new_last = old_last.with_zipped_errors(lle);

      report_lle << "  *err = ";
      new_last.access_by_ref(report_lle, "modelIndex", new_last.params_per_model() - 1);
      report_lle << ";\n";

      layer_params.push_back(std::move(new_last));
    } else {
      report_lle << "  *err = " << lle[0] << ";";
    }
  }

  if (model.cache_fix.has_value()) {
    std::vector<ModelParam> cfv;
    cfv.reserve(model.cache_fix->second.size() * 2);
    for (const auto& [mi, offset] : model.cache_fix->second) {
      cfv.emplace_back(mi);
      cfv.emplace_back(static_cast<uint64_t>(offset));
    }
    layer_params.push_back(LayerParams::create(layer_params.size(), true, 2, std::move(cfv)));
  }

  data_output << "namespace " << namespace_name << " {\n";

  std::vector<std::string> read_code;
  read_code.push_back("bool load(char const* dataPath) {");

  for (const auto& lp : layer_params) {
    if (lp.kind() == LayerParams::Kind::Constant) {
      lp.to_code(data_output);
      continue;
    }

    std::filesystem::path data_path = std::filesystem::path(data_dir) /
                                      (namespace_name + "_" + array_name(lp.index()));
    std::ofstream data_file(data_path, std::ios::binary);
    if (!data_file) {
      throw std::runtime_error("Could not write data file to model data directory");
    }
    lp.write_to(data_file);

    lp.to_decl(data_output);

    read_code.push_back("  {");
    read_code.push_back("    std::ifstream infile(std::filesystem::path(dataPath) / \"" +
                        namespace_name + "_" + array_name(lp.index()) +
                        "\", std::ios::in | std::ios::binary);");
    read_code.push_back("    if (!infile.good()) return false;");
    if (lp.requires_malloc()) {
      read_code.push_back("    " + std::string(lp.pointer_type()) + "* tmp = (" +
                          lp.pointer_type() + "*) malloc(" + std::to_string(lp.size()) + ");");
      read_code.push_back("    " + array_name(lp.index()) + " = tmp;");
      read_code.push_back("    if (" + array_name(lp.index()) + " == NULL) return false;");
    }
    read_code.push_back("    infile.read((char*)" + array_name(lp.index()) + ", " +
                        std::to_string(lp.size()) + ");");
    read_code.push_back("    if (!infile.good()) return false;");
    read_code.push_back("  }");
  }

  read_code.push_back("  return true;");
  read_code.push_back("}");

  std::vector<std::string> free_code;
  free_code.push_back("void cleanup() {");
  for (const auto& lp : layer_params) {
    if (!lp.requires_malloc()) continue;
    free_code.push_back("    free(" + array_name(lp.index()) + ");");
  }
  free_code.push_back("}");

  data_output << "} // namespace\n";

  std::set<StdFunctions> decls;
  std::set<std::string> stdlib_sigs;
  for (const auto& layer : model.model_layers) {
    for (const auto& fn : layer[0]->standard_functions()) {
      decls.insert(fn);
      stdlib_sigs.insert(code(fn));
    }
  }

  code_output << "#include \"" << namespace_name << ".h\"\n";
  code_output << "#include \"" << namespace_name << "_data.h\"\n";
  code_output << "#include <cstdint>\n";
  code_output << "#include <math.h>\n";
  code_output << "#include <cmath>\n";
  code_output << "#include <cstdlib>\n";
  code_output << "#include <fstream>\n";
  code_output << "#include <filesystem>\n";
  code_output << "#include <iostream>\n";
  if (model.cache_fix.has_value()) {
    code_output << "#include <algorithm>\n";
  }

  code_output << "namespace " << namespace_name << " {\n";

  for (const auto& line : read_code) {
    code_output << line << "\n";
  }

  for (const auto& line : free_code) {
    code_output << line << "\n";
  }

  for (const auto& decl : decls) {
    code_output << ::rm_model::decl(decl) << "\n";
  }

  for (const auto& sig : stdlib_sigs) {
    code_output << sig << "\n";
  }

  std::set<std::string> model_sigs;
  for (const auto& layer : model.model_layers) {
    model_sigs.insert(layer[0]->code());
  }

  for (const auto& sig : model_sigs) {
    code_output << sig << "\n";
  }

  code_output << R"(
inline size_t FCLAMP(double inp, double bound) {
  if (inp < 0.0) return 0;
  return (inp > bound ? bound : (size_t)inp);
}

inline double DCLAMP(double inp, double bound) {
  if (inp < 0.0) return 0.0;
  return (inp > bound ? bound : inp);
}
)";

  std::unordered_set<std::string> needed_vars;
  if (model.model_layers.size() > 1) {
    needed_vars.insert("size_t modelIndex;");
  }

  for (const auto& layer : model.model_layers) {
    switch (layer[0]->output_type()) {
      case ModelDataType::Int:
        needed_vars.insert("uint64_t ipred;");
        break;
      case ModelDataType::Float:
        needed_vars.insert("double fpred;");
        break;
      case ModelDataType::Int128:
        needed_vars.insert("uint128_t i128pred;");
        break;
    }
  }

  uint64_t model_bytes = model_size_bytes(model);
  RM_MODEL_LOG_INFO("Generated model size: " << format_bytes(model_bytes) << " (" << model_bytes << " bytes)");

  ModelDataType last_model_output = model.model_layers.back()[0]->output_type();

  auto emit_model_function = [&](const std::string& name,
                                 const std::string& return_type,
                                 const std::string& return_expr,
                                 bool include_err) {
    std::string sig;
    if (include_err) {
      sig = return_type + " " + name + "(" + std::string(c_type(key_type)) + " key, size_t* err)";
    } else {
      sig = return_type + " " + name + "(" + std::string(c_type(key_type)) + " key)";
    }
    code_output << sig << " {\n";
    for (const auto& var : needed_vars) {
      code_output << "  " << var << "\n";
    }
    code_output << "  (void)key;\n";
    if (include_err) {
      code_output << "  (void)err;\n";
    }

    ModelDataType last_output = to_model_data_type(key_type);
    bool needs_check = true;

    for (std::size_t layer_idx = 0; layer_idx < model.model_layers.size(); ++layer_idx) {
      const auto& layer = model.model_layers[layer_idx];
      const auto& layer_param = layer_params[layer_idx];
      ModelDataType required_type = layer[0]->input_type();
      ModelDataType current_output = layer[0]->output_type();

      std::string var_name;
      if (current_output == ModelDataType::Int) var_name = "ipred";
      if (current_output == ModelDataType::Float) var_name = "fpred";
      if (current_output == ModelDataType::Int128) var_name = "i128pred";

      std::size_t num_parameters = layer[0]->params().size();
      if (layer.size() == 1) {
        code_output << "  " << var_name << " = " << layer[0]->function_name() << "(";
        for (std::size_t p = 0; p < num_parameters; ++p) {
          layer_param.access_by_const(code_output, p);
          code_output << ", ";
        }
      } else {
        code_output << "  modelIndex = "
                    << model_index_from_output(last_output, layer.size(), needs_check)
                    << ";\n";
        code_output << "  " << var_name << " = " << layer[0]->function_name() << "(";
        for (std::size_t p = 0; p < num_parameters; ++p) {
          layer_param.access_by_ref(code_output, "modelIndex", p);
          code_output << ", ";
        }
      }

      code_output << "(" << c_type(required_type) << ")key);\n";

      last_output = current_output;
      needs_check = layer[0]->needs_bounds_check();
    }

    if (include_err) {
      if (!report_lle.str().empty()) {
        code_output << report_lle.str() << "\n";
      } else {
        code_output << "  if (err) *err = " << model.num_model_rows - 1 << ";\n";
      }
    }
    code_output << "  return " << return_expr << ";\n";
    code_output << "}\n";
  };

  bool include_err_for_cachefix = model.cache_fix.has_value();
  bool include_err_for_metrics = report_last_layer_errors;
  std::string lookup_name = model.cache_fix.has_value() ? "_rm_model_lookup_pre_cachefix" : "lookup";
  std::string lookup_expr = model_index_from_output(last_model_output, model.num_model_rows, true);
  emit_model_function(lookup_name, "uint64_t", lookup_expr,
                      include_err_for_metrics || include_err_for_cachefix);

  std::string bound = std::to_string(model.num_model_rows) + ".0 - 1.0";
  std::string predict_expr;
  if (last_model_output == ModelDataType::Float) {
    predict_expr = "DCLAMP(fpred, " + bound + ")";
  } else if (last_model_output == ModelDataType::Int) {
    predict_expr = "DCLAMP((double)ipred, " + bound + ")";
  } else {
    predict_expr = "DCLAMP((double)i128pred, " + bound + ")";
  }

  emit_model_function("predict", "double", predict_expr,
                      include_err_for_metrics || include_err_for_cachefix);

  if (model.cache_fix.has_value()) {
    generate_cache_fix_code(code_output, model, array_name(layer_params.size() - 1));
  }

  code_output << "} // namespace\n";

  const int key_type_id = (key_type == KeyType::U64) ? 0 : (key_type == KeyType::U32 ? 1 : 2);
  const bool wrapper_has_err = report_last_layer_errors || model.cache_fix.has_value();

  code_output << "\n#if defined(_WIN32)\n"
              << "#define RM_MODEL_INFER_EXPORT extern \"C\" __declspec(dllexport)\n"
              << "#else\n"
              << "#define RM_MODEL_INFER_EXPORT extern \"C\"\n"
              << "#endif\n\n";

  code_output << "RM_MODEL_INFER_EXPORT const char* rm_model_infer_name() {\n"
              << "  return " << namespace_name << "::NAME;\n"
              << "}\n";
  code_output << "RM_MODEL_INFER_EXPORT bool rm_model_infer_load(char const* dataPath) {\n"
              << "  return " << namespace_name << "::load(dataPath);\n"
              << "}\n";
  code_output << "RM_MODEL_INFER_EXPORT void rm_model_infer_cleanup() {\n"
              << "  " << namespace_name << "::cleanup();\n"
              << "}\n";
  code_output << "RM_MODEL_INFER_EXPORT int rm_model_infer_key_type() {\n"
              << "  return " << key_type_id << ";\n"
              << "}\n";

  const std::string key_cast = "(" + std::string(c_type(key_type)) + ")";
  auto emit_lookup_wrapper = [&](const std::string& name,
                                 const std::string& arg_type,
                                 const std::string& arg_name) {
    code_output << "RM_MODEL_INFER_EXPORT uint64_t " << name << "(" << arg_type
                << " " << arg_name << ", size_t* err) {\n";
    if (wrapper_has_err) {
      code_output << "  return " << namespace_name << "::lookup(" << key_cast
                  << arg_name << ", err);\n";
    } else {
      code_output << "  if (err) *err = 0;\n"
                  << "  return " << namespace_name << "::lookup(" << key_cast
                  << arg_name << ");\n";
    }
    code_output << "}\n";
  };

  emit_lookup_wrapper("rm_model_infer_lookup_u64", "uint64_t", "key");
  emit_lookup_wrapper("rm_model_infer_lookup_u32", "uint32_t", "key");
  emit_lookup_wrapper("rm_model_infer_lookup_f64", "double", "key");

  auto emit_predict_wrapper = [&](const std::string& name,
                                  const std::string& arg_type,
                                  const std::string& arg_name) {
    code_output << "RM_MODEL_INFER_EXPORT double " << name << "(" << arg_type
                << " " << arg_name << ", size_t* err) {\n";
    if (wrapper_has_err) {
      code_output << "  return " << namespace_name << "::predict(" << key_cast
                  << arg_name << ", err);\n";
    } else {
      code_output << "  if (err) *err = 0;\n"
                  << "  return " << namespace_name << "::predict(" << key_cast
                  << arg_name << ");\n";
    }
    code_output << "}\n";
  };

  emit_predict_wrapper("rm_model_infer_predict_u64", "uint64_t", "key");
  emit_predict_wrapper("rm_model_infer_predict_u32", "uint32_t", "key");
  emit_predict_wrapper("rm_model_infer_predict_f64", "double", "key");

  header_output << "#include <cstddef>\n";
  header_output << "#include <cstdint>\n";
  header_output << "namespace " << namespace_name << " {\n";
  header_output << "bool load(char const* dataPath);\n";
  header_output << "void cleanup();\n";
  header_output << "const size_t RM_MODEL_SIZE = " << model_bytes << ";\n";
  header_output << "const uint64_t BUILD_TIME_NS = " << model.build_time_ns << ";\n";
  header_output << "const char NAME[] = \"" << namespace_name << "\";\n";

  if (!model.cache_fix.has_value()) {
    std::string lookup_sig;
    std::string predict_sig;
    if (report_last_layer_errors) {
      lookup_sig = "uint64_t lookup(" + std::string(c_type(key_type)) + " key, size_t* err)";
      predict_sig = "double predict(" + std::string(c_type(key_type)) + " key, size_t* err)";
    } else {
      lookup_sig = "uint64_t lookup(" + std::string(c_type(key_type)) + " key)";
      predict_sig = "double predict(" + std::string(c_type(key_type)) + " key)";
    }
    header_output << lookup_sig << ";\n";
    header_output << predict_sig << ";\n";
  } else {
    header_output << "uint64_t lookup(uint64_t key, size_t* err);\n";
    header_output << "double predict(uint64_t key, size_t* err);\n";
  }

  header_output << "}\n";
}

} // namespace

uint64_t model_size_bytes(const TrainedModel& model) {
  uint64_t total_bytes = 0;
  for (const auto& layer : model.model_layers) {
    std::size_t model_size = 0;
    for (const auto& param : layer[0]->params()) {
      model_size += param.size();
    }
    total_bytes += static_cast<uint64_t>(model_size * layer.size());
  }

  if (!model.last_layer_max_l1s.empty()) {
    total_bytes += static_cast<uint64_t>(model.model_layers.back().size()) * 8ULL;
  }

  if (model.cache_fix.has_value()) {
    total_bytes += static_cast<uint64_t>(model.cache_fix->second.size()) * 16ULL;
  }

  return total_bytes;
}

void emit_model(const std::string& namespace_name,
                TrainedModel trained_model,
                const std::string& output_dir,
                const std::string& data_dir,
                KeyType key_type,
                bool include_errors) {
  if (!include_errors) {
    trained_model.last_layer_max_l1s.clear();
  }

  std::filesystem::path output_root = output_dir.empty()
      ? std::filesystem::path(".")
      : std::filesystem::path(output_dir);
  std::filesystem::create_directories(output_root);

  std::filesystem::path src_dir = output_root / "src";
  std::filesystem::path include_dir = output_root / "include";
  std::filesystem::path lib_dir = output_root / "lib";
  std::filesystem::create_directories(src_dir);
  std::filesystem::create_directories(include_dir);
  std::filesystem::create_directories(lib_dir);

  std::ofstream cpp_file(src_dir / (namespace_name + ".cpp"));
  if (!cpp_file) {
    throw std::runtime_error("Could not write model C++ file");
  }

  std::ofstream data_file(include_dir / (namespace_name + "_data.h"));
  if (!data_file) {
    throw std::runtime_error("Could not write model data header");
  }

  std::ofstream header_file(include_dir / (namespace_name + ".h"));
  if (!header_file) {
    throw std::runtime_error("Could not write model header file");
  }

  generate_code(cpp_file, data_file, header_file, namespace_name, trained_model, data_dir, key_type);

  if (!output_dir.empty()) {
    write_manifest(output_root, src_dir, include_dir, std::filesystem::path(data_dir),
                   namespace_name, trained_model, key_type, include_errors);
  }
}

} // namespace rm_model
