# CLAUDE.md - AI Assistant Guide for DeepGEMM

This document provides comprehensive guidance for AI assistants working with the DeepGEMM codebase.

## Project Overview

**DeepGEMM** is a high-performance GEMM (General Matrix Multiplication) library designed for NVIDIA GPUs, developed by DeepSeek AI. It supports FP8 and BF16 datatypes for both normal and Mix-of-Experts (MoE) grouped scenarios.

### Key Characteristics

- **JIT Compilation**: All kernels are compiled at runtime using a lightweight JIT module - no pre-compilation needed during installation
- **Performance**: Matches or exceeds expert-tuned libraries (up to 1550 TFLOPS on H800)
- **Clean Architecture**: Minimal template reliance; focuses on simplicity and learning
- **GPU Support**: SM90 (Hopper: H100/H800) and SM100 (Blackwell) architectures only
- **Version**: 2.1.1 (as of last update)
- **License**: MIT

### Core Technologies

- **CUDA**: Primary kernel language
- **C++20**: JIT infrastructure and Python bindings
- **Python**: High-level API and testing
- **CUTLASS/CuTe**: Inspiration and some concept usage (minimal template reliance)
- **PyTorch**: Integration and build system

---

## Codebase Structure

### Directory Organization

```
DeepGEMM/
├── .github/                      # CI/CD automation
│   ├── scripts/                  # Build and test scripts
│   └── workflows/                # GitHub Actions (build, publish)
│
├── csrc/                         # C++ source code (~2087 lines)
│   ├── apis/                     # Python binding APIs
│   │   ├── attention.cpp         # MQA attention API
│   │   ├── einsum.cpp           # Einstein summation API
│   │   ├── gemm.cpp             # GEMM API
│   │   ├── layout.cpp           # Layout transformation API
│   │   └── runtime.cpp          # Runtime configuration API
│   │
│   ├── indexing/                # CUDA indexing utilities
│   │   └── main.cu              # Main CUDA implementation
│   │
│   ├── jit/                     # JIT compilation infrastructure
│   │   ├── compiler.hpp         # JIT compiler interface
│   │   ├── cache.hpp            # Kernel cache management
│   │   ├── kernel_runtime.hpp  # Kernel runtime interface
│   │   └── device_runtime.hpp  # Device runtime interface
│   │
│   ├── jit_kernels/             # Kernel implementations
│   │   ├── impls/               # Architecture-specific kernels
│   │   │   ├── sm90_*.hpp       # SM90 (Hopper) kernels
│   │   │   ├── sm100_*.hpp      # SM100 (Blackwell) kernels
│   │   │   └── smxx_*.hpp       # Architecture-agnostic kernels
│   │   └── heuristics/          # Performance tuning
│   │       ├── sm90.hpp         # SM90 heuristics
│   │       ├── sm100.hpp        # SM100 heuristics
│   │       └── common.hpp       # Shared heuristics
│   │
│   ├── utils/                   # C++ utilities
│   │   ├── exception.hpp        # Error handling
│   │   ├── format.hpp           # String formatting
│   │   ├── math.hpp             # Math utilities
│   │   ├── layout.hpp           # Layout utilities
│   │   └── hash.hpp             # Hashing utilities
│   │
│   └── python_api.cpp           # pybind11 module definition
│
├── deep_gemm/                   # Python package
│   ├── __init__.py              # Package initialization, API exports
│   ├── envs.py                  # Generated environment variables (build-time)
│   │
│   ├── include/deep_gemm/       # CUDA kernel headers (~22 files)
│   │   ├── common/              # Shared CUDA utilities
│   │   │   ├── sm90_utils.cuh   # SM90-specific utilities
│   │   │   ├── sm100_utils.cuh  # SM100-specific utilities
│   │   │   ├── scheduler.cuh    # Workload scheduling
│   │   │   ├── reduction.cuh    # Reduction operations
│   │   │   ├── types.hpp        # Type definitions
│   │   │   └── utils.cuh        # General CUDA utilities
│   │   │
│   │   └── impls/               # Actual CUDA kernel implementations
│   │       ├── sm90_fp8_gemm_1d1d.cuh      # FP8 GEMM (1D TMA)
│   │       ├── sm90_fp8_gemm_1d2d.cuh      # FP8 GEMM (2D TMA)
│   │       ├── sm90_bf16_gemm.cuh          # BF16 GEMM
│   │       ├── sm100_fp8_gemm_1d1d.cuh     # SM100 FP8 (1D TMA)
│   │       ├── sm100_fp8_gemm_1d2d.cuh     # SM100 FP8 (2D TMA)
│   │       ├── sm100_bf16_gemm.cuh         # SM100 BF16
│   │       ├── sm90_bmk_bnk_mn.cuh         # K-grouped GEMM
│   │       ├── sm100_bmk_bnk_mn.cuh        # SM100 K-grouped
│   │       └── sm*_*_mqa_logits.cuh        # MQA attention kernels
│   │
│   ├── testing/                 # Testing utilities
│   │   ├── bench.py             # Benchmarking tools
│   │   └── numeric.py           # Numerical validation
│   │
│   └── utils/                   # Python utilities
│       ├── math.py              # Math helpers
│       └── layout.py            # Layout transformations
│
├── tests/                       # Test suite
│   ├── test_fp8.py              # FP8 GEMM tests
│   ├── test_bf16.py             # BF16 GEMM tests
│   ├── test_attention.py        # Attention kernel tests
│   ├── test_layout.py           # Layout transformation tests
│   ├── test_einsum.py           # Einstein summation tests
│   ├── test_lazy_init.py        # Lazy initialization tests
│   └── generators.py            # Test data generators
│
├── third-party/                 # Git submodules
│   ├── cutlass/                 # NVIDIA CUTLASS library
│   └── fmt/                     # {fmt} formatting library
│
├── setup.py                     # Python package setup
├── CMakeLists.txt               # CMake config (IDE indexing ONLY)
├── build.sh                     # Build wheel distribution
├── develop.sh                   # Development setup script
├── install.sh                   # Install script
└── README.md                    # Project documentation
```

### File Counts and Language Breakdown

- **CUDA files (.cu, .cuh)**: ~23 files - Core kernel implementations
- **C++ files (.cpp, .hpp)**: ~36 files - JIT infrastructure, APIs, bindings
- **Python files (.py)**: ~15 files - High-level API, testing, utilities
- **Total source files**: ~74 (excluding third-party)

---

## Development Workflows

### Initial Setup

```bash
# Clone with submodules (REQUIRED)
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
cd DeepGEMM

# Development setup (creates symlinks, builds .so)
./develop.sh

# Run tests
python tests/test_layout.py
python tests/test_attention.py
python tests/test_bf16.py
python tests/test_fp8.py
python tests/test_lazy_init.py
```

### Build System

**IMPORTANT**: The build system has two separate paths:

1. **CMakeLists.txt**: For IDE indexing ONLY - NOT used for actual compilation
2. **setup.py**: Actual build orchestrator using PyTorch's CUDAExtension

#### Build Scripts

- **develop.sh**: Development workflow
  - Links CUTLASS includes to `deep_gemm/include/`
  - Runs `python setup.py build`
  - Creates symlink to built .so file in project root

- **build.sh**: Creates wheel distribution
  - Removes old build artifacts
  - Runs `python setup.py bdist_wheel`

- **install.sh**: Builds and installs via pip
  - Uses `build.sh` output

#### Build Environment Variables

```bash
# Skip CUDA compilation (for CI/testing)
DG_SKIP_CUDA_BUILD=1

# Force building from source (skip pre-compiled wheel download)
DG_FORCE_BUILD=1

# Use local version with git commit hash
DG_USE_LOCAL_VERSION=1  # Default

# Use CUDA runtime API instead of driver API
DG_JIT_USE_RUNTIME_API=0  # Default
```

### JIT Runtime Environment Variables

**General**:
- `DG_JIT_DEBUG`: Print JIT debugging info (default: `0`)
- `DG_PRINT_CONFIGS`: Print selected kernel configs (default: `0`)

**Cache**:
- `DG_JIT_CACHE_DIR`: Cache directory (default: `$HOME/.deep_gemm`)

**Compiler Selection**:
- `DG_JIT_USE_NVRTC`: Use NVRTC instead of NVCC (default: `0`)
  - Faster compilation but may have lower performance in some cases
- `DG_JIT_NVCC_COMPILER`: Specify NVCC path (default: from `torch.utils.cpp_extension.CUDA_HOME`)

**Compiler Options**:
- `DG_JIT_PTXAS_VERBOSE`: Show detailed PTXAS output (default: `0`)
- `DG_JIT_PRINT_COMPILER_COMMAND`: Print NVCC commands (default: `0`)
- `DG_JIT_CPP_STANDARD`: C++ standard version (default: `20`)

### CI/CD Workflows

**GitHub Actions**: `.github/workflows/`

1. **build.yml**: Manual wheel building with parameters
   - Inputs: runs-on, python-version, cuda-version, torch-version, cxx11_abi

2. **publish.yml**: Automated release on version tags (`v*`)
   - Matrix strategy across:
     - Python: 3.9, 3.10, 3.11, 3.12, 3.13
     - PyTorch: 2.5.1, 2.6.0, 2.7.1, 2.8.0
     - CUDA: 12.9.1, 13.0.0
     - C++11 ABI: TRUE/FALSE
   - Uploads wheels to GitHub releases

**Release Process**:
```bash
git tag v2.1.1
git push origin v2.1.1
# GitHub Actions automatically builds and uploads wheels
```

---

## Key Conventions and Patterns

### Naming Conventions

**GEMM Operation**: `D = C + A @ B`

**Function Naming**: `{dtype}_{variant}_gemm_{layout}`
- `dtype`: fp8, bf16
- `variant`: (none), m_grouped, k_grouped
- `layout`: nt (non-transposed × transposed), nn, tn, tt

**Examples**:
- `fp8_gemm_nt`: D = C + A @ B.T (A is non-transposed, B is transposed)
- `m_grouped_fp8_gemm_nt_contiguous`: M-axis grouped FP8 GEMM with contiguous layout
- `k_grouped_fp8_gemm_tn_contiguous`: K-axis grouped FP8 GEMM (for MoE weight backward)

### Layout Conventions

**SM90 (Hopper)**:
- Supports only NT memory layout (row-major A, col-major B)
- Scaling factors: FP32 format
- Requires TMA-aligned and transposed layout for LHS scaling factor

**SM100 (Blackwell)**:
- Supports all memory layouts: NT, TN, NN, TT
- Scaling factors: Packed UE8M0 format (4 values per `torch.int`)
- Same TMA alignment requirements

### TMA (Tensor Memory Accelerator) Dimensions

- **1D1D kernels**: Single TMA dimension (simpler, used in many kernels)
- **1D2D kernels**: Two TMA dimensions (more complex addressing)

### Architecture-Specific Code

**File Prefixes**:
- `sm90_*`: Hopper-specific (H100/H800)
- `sm100_*`: Blackwell-specific
- `smxx_*`: Architecture-agnostic (works on both)

**Compute Capability**:
- SM90: Compute capability 9.0
- SM100: Compute capability 10.0

### Grouped GEMM Concepts

**M-axis Grouping** (for MoE forward/prefill):
- M dimension varies per group
- N and K dimensions fixed
- Each group aligned to GEMM M block size
- Two layout variants:
  - **Contiguous**: Concatenated tokens (training/prefill)
  - **Masked**: Mask-based computation (inference/decoding with CUDA graphs)

**K-axis Grouping** (for MoE weight backward):
- K dimension varies per group
- M and N dimensions fixed
- Function: `k_grouped_fp8_gemm_{tn,nt}_contiguous`

---

## API Patterns

### Entry Points

**Python Package**: `deep_gemm/__init__.py`
- Exports all kernel functions
- Initializes C++ extension module
- Sets up CUDA home detection
- Loads persistent environment variables

**C++ Bindings**: `csrc/python_api.cpp`
- Uses pybind11 for Python/C++ interface
- Registers APIs via namespaces:
  - `deep_gemm::attention::register_apis()`
  - `deep_gemm::einsum::register_apis()`
  - `deep_gemm::gemm::register_apis()`
  - `deep_gemm::layout::register_apis()`
  - `deep_gemm::runtime::register_apis()`

### Main API Categories

**1. FP8 GEMM Kernels**:
```python
# Basic GEMM
fp8_gemm_nt(D, C, A, B, A_sf, B_sf)  # D = C + A @ B.T
fp8_gemm_nn(D, C, A, B, A_sf, B_sf)  # D = C + A @ B (SM100 only)
fp8_gemm_tn(D, C, A, B, A_sf, B_sf)  # D = C + A.T @ B (SM100 only)
fp8_gemm_tt(D, C, A, B, A_sf, B_sf)  # D = C + A.T @ B.T (SM100 only)

# M-grouped (MoE forward)
m_grouped_fp8_gemm_nt_contiguous(D, C, A, B, A_sf, B_sf, group_sizes)
m_grouped_fp8_gemm_nn_contiguous(D, C, A, B, A_sf, B_sf, group_sizes)
m_grouped_fp8_gemm_nt_masked(D, C, A, B, A_sf, B_sf, mask)

# K-grouped (MoE weight backward)
k_grouped_fp8_gemm_nt_contiguous(D, C, A, B, A_sf, B_sf, group_sizes)
k_grouped_fp8_gemm_tn_contiguous(D, C, A, B, A_sf, B_sf, group_sizes)
```

**2. BF16 GEMM Kernels**:
```python
bf16_gemm_nt(D, C, A, B)
bf16_gemm_nn(D, C, A, B)  # SM100 only
bf16_gemm_tn(D, C, A, B)  # SM100 only
bf16_gemm_tt(D, C, A, B)  # SM100 only

# M-grouped
m_grouped_bf16_gemm_nt_contiguous(D, C, A, B, group_sizes)
m_grouped_bf16_gemm_nt_masked(D, C, A, B, mask)
```

**3. Attention Kernels** (V3.2 MQA for DeepSeek indexer):
```python
# Non-paged (prefill)
fp8_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits)

# Paged (decoding)
fp8_paged_mqa_logits(q, k_cache, v_cache, weights, block_tables, ...)
get_paged_mqa_logits_metadata(...)  # Helper for metadata
```

**4. Utility Functions**:
```python
# Runtime configuration
deep_gemm.set_num_sms(num)           # Set max SM count
deep_gemm.get_num_sms()              # Get current max SM count
deep_gemm.set_tc_util(ratio)         # Set tensor core utilization
deep_gemm.get_tc_util()              # Get TC utilization ratio

# Layout utilities
deep_gemm.transform_sf_into_required_layout(sf, ...)
deep_gemm.get_tma_aligned_size(...)
deep_gemm.get_mk_alignment_for_contiguous_layout()
deep_gemm.get_mn_major_tma_aligned_tensor(...)
deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(...)
deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(...)
```

---

## Testing Guidelines

### Test Structure

**Test Files**: `tests/`
- `test_fp8.py`: FP8 GEMM operations (normal, grouped)
- `test_bf16.py`: BF16 GEMM operations
- `test_attention.py`: MQA attention kernels (paged/non-paged)
- `test_layout.py`: Layout transformation tests
- `test_einsum.py`: Einstein summation operations
- `test_lazy_init.py`: Lazy initialization tests
- `generators.py`: Test data generators

### Testing Utilities

**Benchmarking** (`deep_gemm/testing/bench.py`):
```python
bench(func, *args, **kwargs)                # GPU kernel benchmarking
bench_kineto(func, *args, **kwargs)         # PyTorch profiler-based
suppress_stdout_stderr(func)                # Output suppression
```

**Numerical Validation** (`deep_gemm/testing/numeric.py`):
```python
calc_diff(a, b)                             # Cosine similarity
count_bytes(tensor)                         # Memory usage
```

### Testing Best Practices

1. **Validate Correctness**: Always compare against cuBLAS reference
2. **Measure Performance**: Report TFLOPS and GB/s metrics
3. **Test Multiple Shapes**: Various M, N, K dimensions
4. **Test All Layouts**: NT, NN, TN, TT (where supported)
5. **Architecture Testing**: Test on both SM90 and SM100 when available
6. **Grouped GEMM**: Test alignment requirements, multiple group sizes

### Running Tests

```bash
# Individual tests
python tests/test_layout.py
python tests/test_attention.py
python tests/test_bf16.py
python tests/test_fp8.py
python tests/test_lazy_init.py

# All tests (if using pytest)
pytest tests/
```

---

## Performance Considerations

### Key Performance Features

1. **JIT Compilation**: Kernels compiled on first use, cached for reuse
2. **Heuristics**: Architecture-specific performance tuning in `csrc/jit_kernels/heuristics/`
3. **TMA (Tensor Memory Accelerator)**: Hardware-accelerated memory access
4. **Swizzling**: Shared memory optimizations for output
5. **MoE Scheduler**: TMA multicast compatibility
6. **Register Optimization**: Controlled via compiler flags

### Performance Metrics

- **Peak Performance**: Up to 1550 TFLOPS on H800 (FP8)
- **Comparison**: Matches or exceeds expert-tuned libraries
- **Benchmarking**: Use `deep_gemm.testing.bench()` for accurate measurements

### Optimization Guidelines

1. **CUDA Version**: Use 12.9+ for best performance (FFMA interleaving)
2. **TMA Alignment**: Ensure proper alignment for scaling factors
3. **Group Alignment**: M-axis groups must align to block size
4. **SM Count**: Configure via `set_num_sms()` if needed
5. **TC Utilization**: Tune via `set_tc_util()` for specific workloads

---

## Common Pitfalls and Best Practices

### Critical Requirements

**1. Submodule Initialization**:
```bash
# WRONG: git clone git@github.com:deepseek-ai/DeepGEMM.git
# RIGHT:
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git

# If already cloned without --recursive:
git submodule update --init --recursive
```

**2. Architecture Support**:
- **Only SM90 and SM100 are supported**
- No Ampere (SM80) support yet (roadmap item)
- Check GPU compute capability before deployment

**3. CUDA Toolkit Version**:
- SM90: CUDA 12.3+ (12.9+ highly recommended)
- SM100: CUDA 12.9+ required
- Version mismatch can cause compilation failures

**4. CMake Confusion**:
- CMakeLists.txt is for IDE indexing ONLY
- Do NOT use cmake to build the project
- Always use `setup.py` or provided shell scripts

**5. Scaling Factor Format**:
- SM90: FP32 scaling factors
- SM100: Packed UE8M0 format (4 values per int)
- Using wrong format causes incorrect results

### Development Best Practices

**1. IDE Setup**:
- Use CMakeLists.txt for code navigation
- Run `./develop.sh` to set up symlinks and build
- .so file will be symlinked to project root

**2. Testing Changes**:
- Always run relevant tests after modifications
- Compare against cuBLAS baseline
- Check both correctness and performance

**3. JIT Cache**:
- Cache location: `~/.deep_gemm` (or `$DG_JIT_CACHE_DIR`)
- Clear cache if encountering strange errors
- Cache is per-kernel-configuration

**4. Input Validation**:
- Validate tensor layouts before calling kernels
- Check alignment requirements for grouped GEMMs
- Ensure scaling factors have correct TMA alignment

**5. Error Handling**:
- JIT compilation errors appear at runtime, not install time
- Check `DG_JIT_DEBUG=1` for detailed error messages
- Verify CUDA_HOME is set correctly

### Code Modification Guidelines

**When modifying kernels**:
1. Understand JIT compilation: changes take effect after cache invalidation
2. Follow architecture-specific patterns (sm90_* vs sm100_*)
3. Maintain heuristics in `csrc/jit_kernels/heuristics/`
4. Update tests in `tests/` directory

**When adding new APIs**:
1. Add C++ implementation in `csrc/apis/`
2. Register in corresponding `register_apis()` function
3. Add Python exports in `deep_gemm/__init__.py`
4. Create comprehensive tests
5. Document in README.md

**When modifying build system**:
1. Test both development (`develop.sh`) and release (`build.sh`) workflows
2. Verify wheel building with different Python/CUDA/PyTorch versions
3. Update CI/CD workflows if needed
4. Check environment variable handling

---

## Dependencies and Requirements

### System Requirements

**GPU Architecture**:
- NVIDIA SM90 (Hopper: H100, H800)
- NVIDIA SM100 (Blackwell)

**Software**:
- Python: 3.8+
- CUDA Toolkit: 12.3+ (12.9+ highly recommended)
- PyTorch: 2.1+
- C++ Compiler: C++20 support required

**Build Tools**:
- setuptools
- wheel
- torch.utils.cpp_extension

### Third-Party Libraries (Git Submodules)

**CUTLASS** (4.0+):
- NVIDIA's CUDA Templates for Linear Algebra Subroutines
- Location: `third-party/cutlass/`
- Usage: Concepts and some utilities (minimal template usage)

**{fmt}**:
- Modern formatting library for C++
- Location: `third-party/fmt/`
- Usage: String formatting in C++ code

### Python Dependencies

- PyTorch: Deep learning framework (provides CUDA integration)
- NumPy: Numerical operations (indirect via PyTorch)

---

## Debugging and Troubleshooting

### Common Issues

**1. "No module named 'deep_gemm_cpp'"**:
- Cause: .so file not built or not found
- Solution: Run `./develop.sh` or reinstall package

**2. JIT Compilation Failures**:
- Enable debug mode: `DG_JIT_DEBUG=1`
- Print compiler commands: `DG_JIT_PRINT_COMPILER_COMMAND=1`
- Check CUDA_HOME: `echo $CUDA_HOME`
- Verify NVCC: `nvcc --version`

**3. Performance Regression**:
- Check CUDA version (12.9+ recommended)
- Verify kernel config selection: `DG_PRINT_CONFIGS=1`
- Compare NVCC vs NVRTC: toggle `DG_JIT_USE_NVRTC`
- Review heuristics for specific shapes

**4. Numerical Errors**:
- Verify scaling factor format (FP32 vs UE8M0)
- Check TMA alignment
- Validate input tensor layouts
- Compare with cuBLAS reference

**5. Build Failures**:
- Ensure submodules initialized: `git submodule update --init --recursive`
- Check C++20 compiler support
- Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`
- Check CXX11 ABI compatibility

### Debug Environment Setup

```bash
# Maximum debugging output
export DG_JIT_DEBUG=1
export DG_JIT_PRINT_COMPILER_COMMAND=1
export DG_JIT_PTXAS_VERBOSE=1
export DG_PRINT_CONFIGS=1

# Run test
python tests/test_fp8.py
```

---

## Version and Release Information

**Current Version**: 2.1.1

**Versioning Scheme**: Semantic versioning with git revision
- Format: `{major}.{minor}.{patch}+{git_commit_hash}`
- Example: `2.1.1+2f9d878`

**Version File**: `deep_gemm/__init__.py`
```python
__version__ = '2.1.1'
```

**Build-time Version**: Appends git commit hash (when `DG_USE_LOCAL_VERSION=1`)

---

## Project Roadmap

### Completed Features ✓

- More correctness tests for grouped-contiguous layout
- Shared memory swizzling for output
- MoE scheduler with TMA multicast compatibility
- Fix TMA multicast compatibility for indivisible shapes
- Skip useless computation on M
- NVRTC as a faster compiler
- Weight gradient kernels for dense models
- Weight gradient kernels for MoE models
- MMA template refactor with CUTLASS
- Remove shape limitations on N and K
- BF16 kernels

### Pending Features

- Sanitizer for testing
- Better `get_best_configs` modeling
- CUDA PDL support
- Larger TMA multicast size for some shapes
- Split/stream-k optimizations
- Ampere kernels
- Polish docs

---

## Additional Resources

**Documentation**:
- README.md: Primary project documentation
- Test files: Example usage patterns
- Function docstrings: API documentation (in code)

**External References**:
- CUTLASS: https://github.com/nvidia/cutlass
- CuTe: https://github.com/NVIDIA/cutlass/tree/main/include/cute
- UE8M0 Format: https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats

**Related Projects**:
- DeepEP: https://github.com/deepseek-ai/DeepEP (low-latency MoE kernels)

---

## AI Assistant Guidelines

### When Working with This Codebase

**DO**:
- ✓ Always use `./develop.sh` for development setup
- ✓ Run relevant tests after making changes
- ✓ Follow architecture-specific naming (sm90_*, sm100_*, smxx_*)
- ✓ Validate against cuBLAS for correctness
- ✓ Check TMA alignment requirements
- ✓ Use JIT debug flags for troubleshooting
- ✓ Reference test files for usage examples
- ✓ Maintain consistent API patterns

**DON'T**:
- ✗ Use cmake to build the project (CMakeLists.txt is for IDEs only)
- ✗ Assume Ampere support (SM80 not yet supported)
- ✗ Mix scaling factor formats between SM90/SM100
- ✗ Forget to initialize git submodules
- ✗ Modify kernel code without understanding JIT caching
- ✗ Skip performance validation after optimizations
- ✗ Ignore alignment requirements for grouped GEMMs

### Code Search and Navigation Tips

**Finding kernel implementations**:
- Headers: `deep_gemm/include/deep_gemm/impls/`
- JIT wrappers: `csrc/jit_kernels/impls/`

**Finding APIs**:
- Python exports: `deep_gemm/__init__.py`
- C++ bindings: `csrc/apis/`
- Registration: `csrc/python_api.cpp`

**Finding heuristics**:
- Architecture-specific: `csrc/jit_kernels/heuristics/sm{90,100}.hpp`
- Common: `csrc/jit_kernels/heuristics/common.hpp`

**Finding tests**:
- Test files: `tests/test_*.py`
- Test utilities: `deep_gemm/testing/`

---

## License

MIT License - Copyright (c) 2025 DeepSeek

---

*This document was generated for AI assistants to understand and work with the DeepGEMM codebase effectively. Last updated: 2025-11-17*
