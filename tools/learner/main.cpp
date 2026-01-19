#include "load.h"

#include "rm_model/codegen.h"
#include "rm_model/json.h"
#include "rm_model/logging.h"
#include "rm_model/learned_model_selector.h"
#include "rm_model/parallel.h"
#include "rm_model/progress.h"
#include "rm_model/train.h"

#include <atomic>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string input;
  std::optional<std::string> namespace_name;
  std::optional<std::string> models;
  std::optional<uint64_t> branching_factor;
  bool no_code = false;
  std::optional<std::string> param_grid;
  std::optional<std::string> output_dir;
  std::string data_path;
  bool data_path_explicit = false;
  bool no_errors = false;
  std::size_t threads = 4;
  std::optional<std::size_t> bounded;
  std::optional<std::size_t> max_size;
  bool disable_parallel_training = false;
  bool zero_build_time = false;
  std::optional<std::string> optimize;
};

bool is_flag(const std::string& arg) {
  return arg.rfind("-", 0) == 0;
}

using SteadyClock = std::chrono::steady_clock;

double elapsed_ms(SteadyClock::time_point start) {
  auto delta = SteadyClock::now() - start;
  return std::chrono::duration<double, std::milli>(delta).count();
}

std::string format_ms(double ms) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3) << ms << " ms";
  return oss.str();
}

std::string sanitize_component(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  bool last_sep = false;
  for (unsigned char ch : value) {
    if (std::isalnum(ch) || ch == '-' || ch == '_') {
      out.push_back(static_cast<char>(ch));
      last_sep = false;
      continue;
    }
    if (!last_sep) {
      out.push_back('_');
      last_sep = true;
    }
  }
  if (out.empty()) {
    out = "model";
  }
  return out;
}

std::filesystem::path output_root() {
  return std::filesystem::path("rm_model_output");
}

std::filesystem::path auto_output_dir_for_model(const std::string& namespace_name,
                                                const std::string& models,
                                                uint64_t branch_factor) {
  std::ostringstream tag;
  tag << sanitize_component(namespace_name) << "_"
      << (models.empty() ? "auto" : sanitize_component(models))
      << "_bf" << branch_factor;
  return output_root() / tag.str();
}

std::filesystem::path auto_output_dir_for_grid(const std::string& grid_path) {
  auto stem = std::filesystem::path(grid_path).stem().string();
  return output_root() / ("param_grid_" + sanitize_component(stem));
}

std::filesystem::path auto_output_dir_for_selector(const std::string& input_path) {
  auto stem = std::filesystem::path(input_path).stem().string();
  return output_root() / ("selector_" + sanitize_component(stem));
}

struct OutputPaths {
  std::filesystem::path output_dir;
  std::filesystem::path data_dir;
};

OutputPaths resolve_output_paths(const Options& opts,
                                 const std::filesystem::path& default_output_dir) {
  std::filesystem::path output_dir = opts.output_dir.has_value()
      ? std::filesystem::path(*opts.output_dir)
      : default_output_dir;
  std::filesystem::path data_dir = opts.data_path_explicit
      ? std::filesystem::path(opts.data_path)
      : (output_dir / "data");
  return {output_dir, data_dir};
}

Options parse_args(int argc, char** argv) {
  Options opts;
  std::vector<std::string> positional;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (!is_flag(arg)) {
      positional.push_back(arg);
      continue;
    }

    auto require_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++i];
    };

    if (arg == "--no-code") {
      opts.no_code = true;
    } else if (arg == "--param-grid") {
      opts.param_grid = require_value(arg);
    } else if (arg == "--output-dir" || arg == "-o") {
      opts.output_dir = require_value(arg);
    } else if (arg == "--data-path" || arg == "-d") {
      opts.data_path = require_value(arg);
      opts.data_path_explicit = true;
    } else if (arg == "--no-errors") {
      opts.no_errors = true;
    } else if (arg == "--threads" || arg == "-t") {
      opts.threads = std::stoull(require_value(arg));
    } else if (arg == "--bounded") {
      opts.bounded = std::stoull(require_value(arg));
      if (opts.bounded.value() == 0) {
        throw std::runtime_error("Bounded line size must be >= 1");
      }
    } else if (arg == "--max-size") {
      opts.max_size = std::stoull(require_value(arg));
    } else if (arg == "--disable-parallel-training") {
      opts.disable_parallel_training = true;
    } else if (arg == "--zero-build-time") {
      opts.zero_build_time = true;
    } else if (arg == "--optimize") {
      opts.optimize = require_value(arg);
    } else {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }

  if (positional.empty()) {
    throw std::runtime_error("Usage: rm_model_learner <input> [namespace] [models] [branching factor] [options]");
  }

  opts.input = positional[0];
  if (positional.size() > 1) {
    opts.namespace_name = positional[1];
  }
  if (positional.size() > 2) {
    opts.models = positional[2];
  }
  if (positional.size() > 3) {
    opts.branching_factor = std::stoull(positional[3]);
  }

  return opts;
}

std::string namespace_from_path(const std::string& path) {
  auto name = std::filesystem::path(path).filename().string();
  return name.empty() ? std::string("rm_model") : name;
}

template <typename T>
rm_model::TrainedModel train_model_data(rm_model::TrainingData<T>& data,
                                 const Options& opts,
                                 const std::string& models,
                                 uint64_t branch_factor) {
  if (opts.max_size.has_value()) {
    return rm_model::train_for_size<T>(data, *opts.max_size);
  }

  if (opts.bounded.has_value()) {
    if constexpr (std::is_same_v<T, uint64_t>) {
      return rm_model::train_bounded(data, models, branch_factor, *opts.bounded);
    }
    throw std::runtime_error("Can only construct a bounded model on u64 data");
  }

  return rm_model::train<T>(data, models, branch_factor);
}

} // namespace

int main(int argc, char** argv) {
  try {
    rm_model::init_logging();

    Options opts = parse_args(argc, argv);
    rm_model::set_thread_count(opts.threads);

    if (opts.namespace_name.has_value() && opts.param_grid.has_value()) {
      throw std::runtime_error("Can only specify one of namespace or param-grid");
    }

    RM_MODEL_LOG_INFO("Reading " << opts.input << "...");
    auto load_start = SteadyClock::now();

    rm_model::KeyType key_type = rm_model::KeyType::U64;
    DataType dt = DataType::UINT64;
    if (opts.input.find("uint32") != std::string::npos) {
      dt = DataType::UINT32;
      key_type = rm_model::KeyType::U32;
    } else if (opts.input.find("f64") != std::string::npos) {
      dt = DataType::FLOAT64;
      key_type = rm_model::KeyType::F64;
    } else if (opts.input.find("uint64") == std::string::npos) {
      throw std::runtime_error("Data file must contain uint64, uint32, or f64.");
    }

    auto load_result = load_data(opts.input, dt);
    std::size_t num_rows = load_result.first;
    MappedDataset data = std::move(load_result.second);
    RM_MODEL_LOG_INFO("Loaded dataset (" << num_rows << " rows) in " << format_ms(elapsed_ms(load_start)));

    if (opts.optimize.has_value()) {
      auto optimize_start = SteadyClock::now();
      auto results = data.visit([&](auto& typed_data) {
        return rm_model::select_pareto_configs(typed_data, 10);
      });
      RM_MODEL_LOG_INFO("Model selector search completed in " << format_ms(elapsed_ms(optimize_start)));

      rm_model::print_selection_table(results);

      std::string nmspc_prefix = opts.namespace_name.value_or(namespace_from_path(opts.input));
      rm_model::json::Value::Array grid_specs;

      for (std::size_t i = 0; i < results.size(); ++i) {
        auto nmspc = nmspc_prefix + "_" + std::to_string(i);
        grid_specs.push_back(results[i].to_grid_spec(nmspc));
      }

      rm_model::json::Value::Object output_obj;
      output_obj.emplace_back("configs", rm_model::json::Value(std::move(grid_specs)));
      rm_model::json::Value output(std::move(output_obj));

      auto output_paths = resolve_output_paths(opts, auto_output_dir_for_selector(opts.input));
      std::filesystem::create_directories(output_paths.output_dir);
      std::filesystem::path out_path(*opts.optimize);
      if (out_path.is_relative()) {
        out_path = output_paths.output_dir / out_path;
      }

      auto write_start = SteadyClock::now();
      std::ofstream out_file(out_path);
      if (!out_file) {
        throw std::runtime_error("Could not write selector results file");
      }
      rm_model::json::write(out_file, output);
      RM_MODEL_LOG_INFO("Model selector output written in " << format_ms(elapsed_ms(write_start)));
      return 0;
    }

    if (opts.param_grid.has_value()) {
      auto grid_output_paths = resolve_output_paths(opts, auto_output_dir_for_grid(*opts.param_grid));
      std::filesystem::create_directories(grid_output_paths.output_dir);
      auto grid_parse_start = SteadyClock::now();
      auto grid_json = rm_model::json::parse_file(*opts.param_grid);
      const auto* configs_val = grid_json.find("configs");
      if (!configs_val || !configs_val->is_array()) {
        throw std::runtime_error("Configs must have an array as its value");
      }
      RM_MODEL_LOG_INFO("Parsed param grid in " << format_ms(elapsed_ms(grid_parse_start)));

      struct GridConfig {
        std::string layers;
        uint64_t branching_factor;
        std::optional<std::string> namespace_name;
      };

      std::vector<GridConfig> to_test;
      for (const auto& el : configs_val->as_array()) {
        if (!el.is_object()) {
          throw std::runtime_error("Config entry must be an object");
        }
        GridConfig cfg;
        const auto* layers_val = el.find("layers");
        const auto* branch_val = el.find("branching factor");
        if (!layers_val || !layers_val->is_string()) {
          throw std::runtime_error("Config missing layers");
        }
        if (!branch_val || !branch_val->is_number()) {
          throw std::runtime_error("Config missing branching factor");
        }
        cfg.layers = layers_val->as_string();
        cfg.branching_factor = branch_val->as_number().as_uint64();
        if (cfg.branching_factor <= 1) {
          throw std::runtime_error("Branching factor must be >= 2");
        }
        const auto* namespace_val = el.find("namespace");
        if (namespace_val && namespace_val->is_string()) {
          cfg.namespace_name = namespace_val->as_string();
        }
        to_test.push_back(cfg);
      }

      RM_MODEL_LOG_TRACE("# models to train: " << to_test.size());
      rm_model::ProgressBar pbar(to_test.size());

      std::atomic<uint64_t> codegen_ns{0};
      std::atomic<std::size_t> codegen_count{0};

      auto train_one = [&](const GridConfig& cfg) -> rm_model::json::Value {
        RM_MODEL_LOG_TRACE("Training model " << cfg.layers << " with branching factor " << cfg.branching_factor);

        std::ostringstream msg;
        msg << cfg.layers << " bf=" << cfg.branching_factor;
        pbar.update_message(msg.str());

        auto result = data.visit([&](auto& typed_data) {
          auto local_data = typed_data.soft_copy();
          return rm_model::train(local_data, cfg.layers, cfg.branching_factor);
        });

        uint64_t size_bs = rm_model::model_size_bytes(result);

        rm_model::json::Value::Object obj;
        obj.emplace_back("layers", rm_model::json::Value(cfg.layers));
        obj.emplace_back("branching factor", rm_model::json::Value(cfg.branching_factor));
        obj.emplace_back("average error", rm_model::json::Value(result.model_avg_error));
        obj.emplace_back("average error %", rm_model::json::Value(result.model_avg_error / static_cast<double>(num_rows) * 100.0));
        obj.emplace_back("average l2 error", rm_model::json::Value(result.model_avg_l2_error));
        obj.emplace_back("average log2 error", rm_model::json::Value(result.model_avg_log2_error));
        obj.emplace_back("point mae", rm_model::json::Value(result.model_point_mae));
        obj.emplace_back("point rmse", rm_model::json::Value(result.model_point_rmse));
        obj.emplace_back("max error", rm_model::json::Value(result.model_max_error));
        obj.emplace_back("max error %", rm_model::json::Value(result.model_max_error / static_cast<double>(num_rows) * 100.0));
        obj.emplace_back("max log2 error", rm_model::json::Value(result.model_max_log2_error));
        obj.emplace_back("size binary search", rm_model::json::Value(size_bs));
        if (cfg.namespace_name.has_value()) {
          obj.emplace_back("namespace", rm_model::json::Value(*cfg.namespace_name));
        }

        if (opts.zero_build_time) {
          result.build_time_ns = 0;
        }

        if (cfg.namespace_name.has_value()) {
          std::filesystem::path cfg_output_dir = opts.output_dir.has_value()
              ? grid_output_paths.output_dir
              : (grid_output_paths.output_dir / sanitize_component(*cfg.namespace_name));
          auto cfg_paths = resolve_output_paths(opts, cfg_output_dir);
          std::filesystem::create_directories(cfg_paths.output_dir);
          std::filesystem::create_directories(cfg_paths.data_dir);

          auto cg_start = SteadyClock::now();
          rm_model::emit_model(*cfg.namespace_name, std::move(result),
                               cfg_paths.output_dir.string(),
                               cfg_paths.data_dir.string(),
                               key_type, true);
          auto cg_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() - cg_start).count();
          codegen_ns.fetch_add(static_cast<uint64_t>(cg_ns));
          codegen_count.fetch_add(1);
        }

        pbar.inc_with_message(msg.str());
        return rm_model::json::Value(std::move(obj));
      };

      std::vector<rm_model::json::Value> results(to_test.size());
      auto train_start = SteadyClock::now();
      if (opts.disable_parallel_training) {
        RM_MODEL_LOG_TRACE("Training models sequentially");
        for (std::size_t i = 0; i < to_test.size(); ++i) {
          results[i] = train_one(to_test[i]);
        }
      } else {
        RM_MODEL_LOG_TRACE("Training models in parallel");
        rm_model::parallel_for(to_test.size(), [&](std::size_t idx) {
          results[idx] = train_one(to_test[idx]);
        });
      }
      pbar.finish();
      RM_MODEL_LOG_INFO("Param-grid training completed in " << format_ms(elapsed_ms(train_start)));
      if (codegen_count.load() > 0) {
        double cg_ms = static_cast<double>(codegen_ns.load()) / 1e6;
        RM_MODEL_LOG_INFO("Param-grid codegen (" << codegen_count.load() << " models) completed in "
                                             << format_ms(cg_ms));
      }

      auto write_start = SteadyClock::now();
      std::filesystem::path grid_file = std::filesystem::path(*opts.param_grid).filename();
      std::filesystem::path out_path = grid_output_paths.output_dir / (grid_file.string() + "_results");
      std::ofstream out_file(out_path);
      if (!out_file) {
        throw std::runtime_error("Could not write results file");
      }
      rm_model::json::Value::Object out_obj;
      out_obj.emplace_back("results", rm_model::json::Value(std::move(results)));
      rm_model::json::Value out(std::move(out_obj));
      rm_model::json::write(out_file, out);
      RM_MODEL_LOG_INFO("Param-grid results written in " << format_ms(elapsed_ms(write_start)));
      return 0;
    }

    if (opts.namespace_name.has_value()) {
      std::string models = opts.models.value_or("");
      if (models.empty() && !opts.max_size.has_value()) {
        throw std::runtime_error("Model specification required");
      }

      uint64_t branch_factor = opts.branching_factor.value_or(0);
      if (branch_factor == 0 && !opts.max_size.has_value()) {
        throw std::runtime_error("Branching factor required");
      }
      if (branch_factor <= 1 && !opts.max_size.has_value()) {
        throw std::runtime_error("Branching factor must be >= 2");
      }

      auto train_start = SteadyClock::now();
      auto trained_model = data.visit([&](auto& typed_data) {
        auto local_data = typed_data.soft_copy();
        return train_model_data(local_data, opts, models, branch_factor);
      });
      RM_MODEL_LOG_INFO("Model training completed in " << format_ms(elapsed_ms(train_start)));

      bool no_errors = opts.no_errors;
      RM_MODEL_LOG_INFO("Model build time: " << trained_model.build_time_ns / 1000000 << " ms");
      RM_MODEL_LOG_INFO("Average model error: " << trained_model.model_avg_error << " ("
                                           << trained_model.model_avg_error / static_cast<double>(num_rows) * 100.0
                                           << "%)");
      RM_MODEL_LOG_INFO("Average model L2 error: " << trained_model.model_avg_l2_error);
      RM_MODEL_LOG_INFO("Average model log2 error: " << trained_model.model_avg_log2_error);
      RM_MODEL_LOG_INFO("Point MAE: " << trained_model.model_point_mae << " ("
                                 << trained_model.model_point_mae / static_cast<double>(num_rows) * 100.0
                                 << "%)");
      RM_MODEL_LOG_INFO("Point RMSE: " << trained_model.model_point_rmse << " ("
                                  << trained_model.model_point_rmse / static_cast<double>(num_rows) * 100.0
                                  << "%)");
      RM_MODEL_LOG_INFO("Max model log2 error: " << trained_model.model_max_log2_error);
      RM_MODEL_LOG_INFO("Max model error on model " << trained_model.model_max_error_idx
                                              << ": " << trained_model.model_max_error << " ("
                                              << trained_model.model_max_error / static_cast<double>(num_rows) * 100.0
                                              << "%)");

      if (!opts.no_code) {
        auto output_paths = resolve_output_paths(
            opts,
            auto_output_dir_for_model(*opts.namespace_name,
                                      trained_model.model_spec,
                                      trained_model.branching_factor));
        std::filesystem::create_directories(output_paths.output_dir);
        std::filesystem::create_directories(output_paths.data_dir);

        if (opts.zero_build_time) {
          trained_model.build_time_ns = 0;
        }
        auto codegen_start = SteadyClock::now();
        rm_model::emit_model(*opts.namespace_name, std::move(trained_model),
                             output_paths.output_dir.string(),
                             output_paths.data_dir.string(),
                             key_type, !no_errors);
        RM_MODEL_LOG_INFO("Code generation completed in " << format_ms(elapsed_ms(codegen_start)));
      } else {
        RM_MODEL_LOG_TRACE("Skipping code generation due to CLI flag");
      }
      return 0;
    }

    RM_MODEL_LOG_TRACE("Must specify either a namespace or a parameter grid");
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
