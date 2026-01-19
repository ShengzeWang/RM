## RM_Model

RM_Model is a standalone **C++** library for **low-overhead, high-performance training and inference** of **multi-layer learned models** for computer networks and systems. RM Model serves as one of the **foundational learning modules** for our **LearnHash** library: https://github.com/ShengzeWang/LearnedHash. It can also be used independently as a **learned index**, retaining an **RMI-like interface**.


It is a fresh C++ implementation inspired by the Recursive Model Index (RMI) Rust implementation from the Learned Systems project: https://github.com/learnedsystems/RMI. RM Model preserves the core ideas of RMI while delivering **improvements in performance (model building, selecting and inference), usability, and extensibility** for broader applications.


RM Model is designed to be **fast, lightweight, and easy to embed** in other projects.

## Citation

If you use RM Model in your research or in a published system, please cite the following papers and the original Recursive Model Index (RMI) papers.
### BibTeX

```bibtex
@INPROCEEDINGS{11192384,
  author={Wang, Shengze and Liu, Yi and Zhang, Xiaoxue and Hu, Liting and Qian, Chen},
  booktitle={2025 IEEE 33rd International Conference on Network Protocols (ICNP)}, 
  title={A Distributed Learned Hash Table}, 
  year={2025},
  pages={1-11},
  doi={10.1109/ICNP65844.2025.11192384}
}

@INPROCEEDINGS{10858529,
  author={Wang, Shengze and Liu, Yi and Zhang, Xiaoxue and Hu, Liting and Qian, Chen},
  booktitle={2024 IEEE 32nd International Conference on Network Protocols (ICNP)}, 
  title={Poster: Distributed Learned Hash Table}, 
  year={2024},
  pages={1-2},
  doi={10.1109/ICNP61940.2024.10858529}
}

@INPROCEEDINGS{11192399,
  author={Wang, Shengze and Liu, Yi and Qian, Chen},
  booktitle={2025 IEEE 33rd International Conference on Network Protocols (ICNP)}, 
  title={Poster: Vortex: Efficient Decentralized Vector Overlay for Similarity Search and Delivery}, 
  year={2025},
  pages={1-3},
  doi={10.1109/ICNP65844.2025.11192399}
}
```

## Build

Requirements:

- CMake 3.16+
- C++17 compiler

From this directory:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

From the repository root:

```bash
cmake -S rm_model -B rm_model/build -DCMAKE_BUILD_TYPE=Release
cmake --build rm_model/build --config Release
```

Binaries:

- `build/rm_model_learner` (training + codegen + selector)
- `build/rm_model_inferencer` (runtime inference)

## Quick Start

Train a two-layer model and emit code:

```bash
build/rm_model_learner dataset/osm_cellids_200M_uint64 my_model linear,linear 100
```

Multi-layer example:

```bash
build/rm_model_learner dataset/osm_cellids_200M_uint64 my_model linear,linear,linear 100
```

## Output Layout

- If `--output-dir` is not set, outputs go under `rm_model_output/<namespace>_<models>_bf<branch>`.
- Generated artifacts live in the output directory, and parameter blobs are written to `data/`
  unless `--data-path` is provided.

Generated artifacts:

- `src/<namespace>.cpp`: lookup implementation
- `include/<namespace>.h`: public interface
- `include/<namespace>_data.h`: data declarations and a `load()` helper
- `data/`: binary parameter blobs (for example, `<namespace>_L1_PARAMETERS`)
- `model.json`: manifest for downstream tooling
- `lib/` (optional): inferencer auto-build output

`model.json` includes the model spec, key type, and a `paths` section pointing at `src/`,
`include/`, `data/`, and `lib/`. The inferencer uses it when present.

## Inferencer

The inferencer loads a compiled model shared library plus its `data/` blobs and runs lookups on
either a dataset or a single key. If the shared library is missing, it will build one from the
generated `src/<namespace>.cpp` automatically.

Notes on auto-build:

- Compiler selection: `RM_MODEL_CXX` (preferred) or `CXX`; defaults to `c++` on Unix and `cl` on Windows.
- Outputs: `lib/<namespace>.so` (Linux), `lib/<namespace>.dylib` (macOS), `lib/<namespace>.dll` (Windows).

Manual build (optional):

- Linux:
  ```bash
  g++ -std=c++17 -O3 -shared -fPIC -I include -o lib/libmy_model.so src/my_model.cpp
  ```
- macOS:
  ```bash
  c++ -std=c++17 -O3 -dynamiclib -I include -o lib/libmy_model.dylib src/my_model.cpp
  ```
- Windows (Developer Command Prompt):
  ```cmd
  cl /LD /I include src\\my_model.cpp /Fe:lib\\my_model.dll
  ```

Run the inferencer on a dataset:

```bash
build/rm_model_inferencer --model-dir rm_model_output/my_model_linear_bf100 \
  --dataset dataset/osm_cellids_200M_uint64 --out predictions.csv
```

Floating prediction example:

```bash
build/rm_model_inferencer --model-dir rm_model_output/my_model_linear_bf100 \
  --dataset dataset/osm_cellids_200M_uint64 --pred-precision double --out predictions.csv
```

Single-key example:

```bash
build/rm_model_inferencer --model-dir rm_model_output/my_model_linear_bf100 \
  --key 12345 --key-type u64
```

Notes:

- Use `--model-lib` if the shared library is not in the model directory or has a custom name.
- If you pass `--model-lib` without `--model-dir`, also pass `--data-dir`.
- Dataset type defaults to the model key type; you can override with `--key-type`.
- Use `--pred-precision double` to return floating predictions; default is integer lookup.

## Input Data Format

Binary file layout:

- first 8 bytes: `uint64_t` count in little-endian
- payload: `count` items, each item in little-endian
  - `uint64_t`, `uint32_t`, or `double` (IEEE 754)

Type inference is based on the input path:

- contains `uint64` -> `uint64_t`
- contains `uint32` -> `uint32_t`
- contains `f64` -> `double`

If no marker is found, the program exits with an error.

## Learner CLI Usage

```bash
build/rm_model_learner <input> [namespace] [models] [branching factor] [options]
```

Notes:

- `branching factor` must be `>= 2`.

Key flags:

- `--optimize <file>`: run model selector and write a JSON grid.
- `--param-grid <file>`: train configs from a JSON file and emit `<file>_results`.
- `--output-dir <dir>`: base output directory for generated code, headers, and data.
- `--data-path <dir>`: output directory for model parameter binaries (defaults to `<output-dir>/data`).
- `--no-code`: skip code generation.
- `--no-errors`: omit last-layer errors and change lookup signature.
- `--threads <count>`: number of threads for parallel training (default: 4).
- `--bounded <line_size>`: build a cache-fix bounded model (only for `uint64` input).
- `--max-size <bytes>`: choose a config smaller than the specified size.
- `--disable-parallel-training`: force sequential training of param-grid configs.
- `--zero-build-time`: zero out the `BUILD_TIME_NS` field for reproducible outputs.

Environment:

- `RM_MODEL_SELECTOR_PROFILE`: selector bias (`fast`, `memory`, `disk`).
- `RM_MODEL_SELECTOR_LAYERS`: selector search depth (>=2, default 2).
- `RM_MODEL_SELECTOR_LIGHT_LAYERS`: when enabled, restricts non-top layers to light models
  (currently `linear`) so only the top layer can use heavier models. Defaults to `1` when
  `RM_MODEL_SELECTOR_LAYERS > 2`.
- `RM_MODEL_SELECTOR_JOBS`: limit concurrent selector evaluations (defaults to `1` when layers > 2,
  otherwise uses `--threads`).
- `RM_MODEL_SELECTOR_MAX_LEAF_MODELS`: skip configs that would create more leaf models than this
  limit (default `1000000` for layers > 2, `10000000` otherwise).

## Supported Models

Model names for `--models` / param-grid JSON:

- `linear`
- `robust_linear`
- `linear_spline`
- `cubic`
- `loglinear`
- `normal`
- `lognormal`
- `radix`
- `radix8`
- `radix18`
- `radix22`
- `radix26`
- `radix28`
- `bradix`
- `histogram`

Notes:

- Some models are restricted to the top layer (`radix*`, `bradix`, `histogram`).
- Two-layer configs are the common fast path; multi-layer configs are supported and use the
  same codegen pipeline.

## Model Selector Usage

Run selector search and write a config grid:

```bash
build/rm_model_learner dataset/osm_cellids_200M_uint64 --optimize selector_out.json --threads 4
```

Then train configs from the grid:

```bash
build/rm_model_learner dataset/osm_cellids_200M_uint64 --param-grid selector_out.json --threads 4
```

Deeper search with light-layer restriction:

```bash
RM_MODEL_SELECTOR_LAYERS=3 RM_MODEL_SELECTOR_LIGHT_LAYERS=1 \
  build/rm_model_learner dataset/osm_cellids_200M_uint64 --optimize selector_out_3layer.json
```

## Param-Grid JSON

Example:

```json
{
  "configs": [
    {
      "layers": "linear,linear",
      "branching factor": 100,
      "namespace": "my_model_0"
    }
  ]
}
```

Notes:

- `branching factor` must be an integer JSON number.
- `namespace` is optional; if present, codegen runs for that config.
- Use comma-separated model names in `layers` for multi-layer configs.

## Runtime Usage (Generated Model)

The generated model exposes:

```cpp
namespace my_model {
  bool load(char const* dataPath);
  void cleanup();
  const size_t RM_MODEL_SIZE = ...;
  const uint64_t BUILD_TIME_NS = ...;
  const char NAME[] = "my_model";
  uint64_t lookup(uint64_t key, size_t* err); // or lookup(uint64_t key)
  double predict(uint64_t key, size_t* err); // or predict(uint64_t key)
}
```

Runtime usage:

1. call `load(dataPath)` once
2. call `lookup(...)` for integer predictions or `predict(...)` for floating predictions
3. call `cleanup()` if parameters were heap-allocated

## Library Integration

CMake (recommended):

```cmake
add_subdirectory(rm_model)
target_link_libraries(your_target PRIVATE rm_model)
```

Example usage:

```cpp
#include "rm_model/train.h"
#include "rm_model/training_data.h"

rm_model::TrainingData<uint64_t> data = ...;
auto trained = rm_model::train(data, "linear,linear", 100);
```

## Training Metrics

The CLI prints both bound-style and point-wise metrics:

- average model error / log2 error: weighted per-leaf max error (bound-focused)
- average model L2 error: weighted squared per-leaf max error
- point MAE / point RMSE: mean and root-mean-square per-key absolute error

## Project Layout (Core Files)

- `tools/learner/main.cpp`: learner CLI entry point and orchestration.
- `tools/learner/load.h` / `tools/learner/load.cpp`: memory-mapped data loader + type dispatch.
- `tools/inferencer/main.cpp`: inferencer CLI for dynamic model loading.
- `include/rm_model/train.h` / `src/rm_model/train.cpp`: training entry points.
- `src/rm_model/train_layer.cpp`: two-layer and multi-layer trainers.
- `include/rm_model/codegen.h` / `src/rm_model/codegen.cpp`: code + data emission.
- `include/rm_model/learned_model_selector.h` / `src/rm_model/learned_model_selector.cpp`: model selector.
- `include/rm_model/json.h` / `src/rm_model/json.cpp`: minimal JSON parser/writer.
- `include/rm_model/training_data.h`: training data abstraction + iterators.
- `include/rm_model/parallel.h` / `src/rm_model/parallel.cpp`: thread pool + parallel loops.
- `include/rm_model/models/*` and `src/rm_model/models/*`: model implementations.

## Performance Notes

- Prefer Release builds.
- Use `--threads` to match physical cores; for very large data, a single thread may be faster
  for memory-bound training.
- Multi-layer training computes conservative error bounds using lower-bound correction; this
  adds one extra scan and per-leaf metadata proportional to layer width.

## Development Notes

- Training entry point: `src/rm_model/train.cpp` dispatches to `src/rm_model/train_layer.cpp` for
  two-layer or multi-layer specs.
- Model additions: implement a new `Model` under `include/rm_model/models` and
  `src/rm_model/models`, then register it in `src/rm_model/train.cpp` and
  `src/rm_model/train_layer.cpp`.
- Keep `TrainingData` providers zero-copy when possible; they must return sorted keys and
  consistent offsets.
- `rm_model/json.*` is intentionally minimal; avoid lax parsing or float-to-int coercion.
- Generated code uses packed structs for cache-fix; keep MSVC compatibility.

## License

This project is licensed under the **Apache License, Version 2.0**.
