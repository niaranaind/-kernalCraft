//
// MiniPTX Runtime Loader: executes emitted PTX on NVIDIA GPUs via the CUDA Driver API.
//
// Target architecture: NVIDIA Tesla V100 (sm_70) on DGX.
//
// Pipeline:
//   cuInit -> cuDeviceGet -> cuCtxCreate -> cuModuleLoadDataEx -> cuModuleGetFunction
//   -> cuMemAlloc -> cuMemcpyHtoD -> cuLaunchKernel -> cuMemcpyDtoH -> CPU verification.
//

#include <cuda.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Error Checking Macro
// ---------------------------------------------------------------------------

#define CUDA_CHECK(expr)                                                    \
    do {                                                                    \
        CUresult err = (expr);                                              \
        if (err != CUDA_SUCCESS) {                                          \
            const char* err_name = nullptr;                                 \
            const char* err_str = nullptr;                                  \
            cuGetErrorName(err, &err_name);                                 \
            cuGetErrorString(err, &err_str);                                \
            std::cerr << "CUDA Driver API error at " << __FILE__ << ":"     \
                      << __LINE__ << " in " << #expr << ": "                \
                      << (err_name ? err_name : "UNKNOWN") << " ("          \
                      << (err_str ? err_str : "") << ")\n";                 \
            return false;                                                   \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Float Tolerance Comparison
// ---------------------------------------------------------------------------

// Compares two floats using both absolute and relative tolerance
// abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))
static inline bool float_close(float a, float b, float atol = 1e-5f, float rtol = 1e-4f) {
    if (std::isnan(a) || std::isnan(b)) return false;
    float diff = std::fabs(a - b);
    float max_abs = std::max(std::fabs(a), std::fabs(b));
    return diff <= std::max(atol, rtol * max_abs);
}

// ---------------------------------------------------------------------------
// Module Loading with JIT Log Buffer
// ---------------------------------------------------------------------------

static bool load_ptx_module(const std::string& ptx_source, CUmodule* out_module) {
    char error_log[8192] = {0};
    char info_log[8192] = {0};

    const unsigned int num_options = 4;
    CUjit_option options[num_options];
    void* option_values[num_options];

    // Setup JIT compilation error and info log buffers
    options[0] = CU_JIT_ERROR_LOG_BUFFER;
    option_values[0] = reinterpret_cast<void*>(error_log);

    options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    option_values[1] = reinterpret_cast<void*>(uintptr_t(sizeof(error_log)));

    options[2] = CU_JIT_INFO_LOG_BUFFER;
    option_values[2] = reinterpret_cast<void*>(info_log);

    options[3] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    option_values[3] = reinterpret_cast<void*>(uintptr_t(sizeof(info_log)));

    CUresult err = cuModuleLoadDataEx(out_module, ptx_source.c_str(), num_options, options, option_values);
    if (err != CUDA_SUCCESS) {
        const char* err_name = nullptr;
        cuGetErrorName(err, &err_name);
        std::cerr << "cuModuleLoadDataEx failed (" << (err_name ? err_name : "UNKNOWN") << "):\n";
        if (error_log[0] != '\0') {
            std::cerr << "--- JIT Error Log ---\n" << error_log << "\n";
        }
        return false;
    }
    return true;
}

static std::string read_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "error: cannot open file '" << path << "'\n";
        return "";
    }
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// ---------------------------------------------------------------------------
// Kernel Test 1: vecadd
// ---------------------------------------------------------------------------

bool test_kernel_vecadd(CUcontext ctx, const std::string& ptx_path) {
    std::string ptx = read_file(ptx_path);
    if (ptx.empty()) return false;

    CUmodule module;
    if (!load_ptx_module(ptx, &module)) return false;

    CUfunction kernel;
    CUDA_CHECK(cuModuleGetFunction(&kernel, module, "vecadd"));

    // n = 1003 (not a multiple of block size 256; tests bounds check)
    const int n = 1003;
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    std::vector<int> h_a(n), h_b(n), h_c(n, 0), h_ref(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = i * 3 + 1;
        h_b[i] = i * 7 - 5;
        h_ref[i] = h_a[i] + h_b[i];
    }

    CUdeviceptr d_a, d_b, d_c;
    size_t bytes = n * sizeof(int);
    CUDA_CHECK(cuMemAlloc(&d_a, bytes));
    CUDA_CHECK(cuMemAlloc(&d_b, bytes));
    CUDA_CHECK(cuMemAlloc(&d_c, bytes));

    CUDA_CHECK(cuMemcpyHtoD(d_a, h_a.data(), bytes));
    CUDA_CHECK(cuMemcpyHtoD(d_b, h_b.data(), bytes));

    void* args[] = { &d_a, &d_b, &d_c, const_cast<int*>(&n) };
    CUDA_CHECK(cuLaunchKernel(kernel, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, args, nullptr));
    CUDA_CHECK(cuCtxSynchronize());

    CUDA_CHECK(cuMemcpyDtoH(h_c.data(), d_c, bytes));

    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_c);
    cuModuleUnload(module);

    for (int i = 0; i < n; ++i) {
        if (h_c[i] != h_ref[i]) {
            std::cerr << "vecadd mismatch at index " << i << ": expected " << h_ref[i] << ", got " << h_c[i] << "\n";
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Kernel Test 2: saxpy
// ---------------------------------------------------------------------------

bool test_kernel_saxpy(CUcontext ctx, const std::string& ptx_path) {
    std::string ptx = read_file(ptx_path);
    if (ptx.empty()) return false;

    CUmodule module;
    if (!load_ptx_module(ptx, &module)) return false;

    CUfunction kernel;
    CUDA_CHECK(cuModuleGetFunction(&kernel, module, "saxpy"));

    const int n = 1003; // non-multiple of 256
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    const float a = 2.5f;

    std::vector<float> h_x(n), h_y(n), h_ref(n);
    for (int i = 0; i < n; ++i) {
        h_x[i] = static_cast<float>(i) * 0.5f;
        h_y[i] = static_cast<float>(i) * 1.2f - 4.0f;
        h_ref[i] = a * h_x[i] + h_y[i];
    }

    CUdeviceptr d_x, d_y;
    size_t bytes = n * sizeof(float);
    CUDA_CHECK(cuMemAlloc(&d_x, bytes));
    CUDA_CHECK(cuMemAlloc(&d_y, bytes));

    CUDA_CHECK(cuMemcpyHtoD(d_x, h_x.data(), bytes));
    CUDA_CHECK(cuMemcpyHtoD(d_y, h_y.data(), bytes));

    void* args[] = { const_cast<float*>(&a), &d_x, &d_y, const_cast<int*>(&n) };
    CUDA_CHECK(cuLaunchKernel(kernel, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, args, nullptr));
    CUDA_CHECK(cuCtxSynchronize());

    CUDA_CHECK(cuMemcpyDtoH(h_y.data(), d_y, bytes));

    cuMemFree(d_x);
    cuMemFree(d_y);
    cuModuleUnload(module);

    for (int i = 0; i < n; ++i) {
        if (!float_close(h_y[i], h_ref[i])) {
            std::cerr << "saxpy mismatch at index " << i << ": expected " << h_ref[i] << ", got " << h_y[i] << "\n";
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Kernel Test 3: scale
// ---------------------------------------------------------------------------

bool test_kernel_scale(CUcontext ctx, const std::string& ptx_path) {
    std::string ptx = read_file(ptx_path);
    if (ptx.empty()) return false;

    CUmodule module;
    if (!load_ptx_module(ptx, &module)) return false;

    CUfunction kernel;
    CUDA_CHECK(cuModuleGetFunction(&kernel, module, "scale"));

    const int n = 1003;
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    const float s = 1.75f;

    std::vector<float> h_x(n), h_ref(n);
    for (int i = 0; i < n; ++i) {
        h_x[i] = static_cast<float>(i + 1) * 0.25f;
        h_ref[i] = h_x[i] * s;
    }

    CUdeviceptr d_x;
    size_t bytes = n * sizeof(float);
    CUDA_CHECK(cuMemAlloc(&d_x, bytes));
    CUDA_CHECK(cuMemcpyHtoD(d_x, h_x.data(), bytes));

    void* args[] = { &d_x, const_cast<float*>(&s), const_cast<int*>(&n) };
    CUDA_CHECK(cuLaunchKernel(kernel, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, args, nullptr));
    CUDA_CHECK(cuCtxSynchronize());

    CUDA_CHECK(cuMemcpyDtoH(h_x.data(), d_x, bytes));

    cuMemFree(d_x);
    cuModuleUnload(module);

    for (int i = 0; i < n; ++i) {
        if (!float_close(h_x[i], h_ref[i])) {
            std::cerr << "scale mismatch at index " << i << ": expected " << h_ref[i] << ", got " << h_x[i] << "\n";
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Kernel Test 4: ewmax
// ---------------------------------------------------------------------------

bool test_kernel_ewmax(CUcontext ctx, const std::string& ptx_path) {
    std::string ptx = read_file(ptx_path);
    if (ptx.empty()) return false;

    CUmodule module;
    if (!load_ptx_module(ptx, &module)) return false;

    CUfunction kernel;
    CUDA_CHECK(cuModuleGetFunction(&kernel, module, "ewmax"));

    const int n = 1003;
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    std::vector<float> h_a(n), h_b(n), h_c(n, 0.0f), h_ref(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i % 17) - 8.0f;
        h_b[i] = static_cast<float>((i * 7) % 19) - 9.0f;
        h_ref[i] = (h_a[i] > h_b[i]) ? h_a[i] : h_b[i];
    }

    CUdeviceptr d_a, d_b, d_c;
    size_t bytes = n * sizeof(float);
    CUDA_CHECK(cuMemAlloc(&d_a, bytes));
    CUDA_CHECK(cuMemAlloc(&d_b, bytes));
    CUDA_CHECK(cuMemAlloc(&d_c, bytes));

    CUDA_CHECK(cuMemcpyHtoD(d_a, h_a.data(), bytes));
    CUDA_CHECK(cuMemcpyHtoD(d_b, h_b.data(), bytes));

    void* args[] = { &d_a, &d_b, &d_c, const_cast<int*>(&n) };
    CUDA_CHECK(cuLaunchKernel(kernel, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, args, nullptr));
    CUDA_CHECK(cuCtxSynchronize());

    CUDA_CHECK(cuMemcpyDtoH(h_c.data(), d_c, bytes));

    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_c);
    cuModuleUnload(module);

    for (int i = 0; i < n; ++i) {
        if (!float_close(h_c[i], h_ref[i])) {
            std::cerr << "ewmax mismatch at index " << i << ": expected " << h_ref[i] << ", got " << h_c[i] << "\n";
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Kernel Test 5: fill (tests sequential for-loop in kernel)
// ---------------------------------------------------------------------------

bool test_kernel_fill(CUcontext ctx, const std::string& ptx_path) {
    std::string ptx = read_file(ptx_path);
    if (ptx.empty()) return false;

    CUmodule module;
    if (!load_ptx_module(ptx, &module)) return false;

    CUfunction kernel;
    CUDA_CHECK(cuModuleGetFunction(&kernel, module, "fill"));

    const int n = 500;
    std::vector<int> h_x(n, 0), h_ref(n);
    for (int i = 0; i < n; ++i) {
        h_ref[i] = i * 2;
    }

    CUdeviceptr d_x;
    size_t bytes = n * sizeof(int);
    CUDA_CHECK(cuMemAlloc(&d_x, bytes));

    // Launched with 1 block, 1 thread since the kernel executes a for loop over all n
    void* args[] = { &d_x, const_cast<int*>(&n) };
    CUDA_CHECK(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
    CUDA_CHECK(cuCtxSynchronize());

    CUDA_CHECK(cuMemcpyDtoH(h_x.data(), d_x, bytes));

    cuMemFree(d_x);
    cuModuleUnload(module);

    for (int i = 0; i < n; ++i) {
        if (h_x[i] != h_ref[i]) {
            std::cerr << "fill mismatch at index " << i << ": expected " << h_ref[i] << ", got " << h_x[i] << "\n";
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Main Driver
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    std::string ptx_dir = (argc > 1) ? argv[1] : "build";

    std::cout << "========================================================\n";
    std::cout << " MiniPTX GPU Runtime Verification (CUDA Driver API)     \n";
    std::cout << " Target: Tesla V100 (sm_70)                             \n";
    std::cout << "========================================================\n\n";

    CUDA_CHECK(cuInit(0));

    int device_count = 0;
    CUDA_CHECK(cuDeviceGetCount(&device_count));
    if (device_count == 0) {
        std::cerr << "error: no CUDA-capable devices found.\n";
        return 1;
    }

    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));

    char dev_name[256];
    CUDA_CHECK(cuDeviceGetName(dev_name, sizeof(dev_name), device));
    std::cout << "Using GPU Device 0: " << dev_name << "\n\n";

    CUcontext context;
    CUDA_CHECK(cuCtxCreate(&context, 0, device));

    struct TestCase {
        std::string name;
        std::string ptx_file;
        bool (*func)(CUcontext, const std::string&);
    };

    std::vector<TestCase> cases = {
        {"vecadd",  ptx_dir + "/vecadd.ptx",  test_kernel_vecadd},
        {"saxpy",   ptx_dir + "/saxpy.ptx",   test_kernel_saxpy},
        {"scale",   ptx_dir + "/scale.ptx",   test_kernel_scale},
        {"ewmax",   ptx_dir + "/ewmax.ptx",   test_kernel_ewmax},
        {"fill",    ptx_dir + "/fill.ptx",    test_kernel_fill}
    };

    int passed = 0;
    for (const auto& tc : cases) {
        std::cout << "Running test: " << tc.name << " (" << tc.ptx_file << ")... ";
        if (tc.func(context, tc.ptx_file)) {
            std::cout << "[PASS]\n";
            ++passed;
        } else {
            std::cout << "[FAIL]\n";
        }
    }

    CUDA_CHECK(cuCtxDestroy(context));

    std::cout << "\nResult: " << passed << "/" << cases.size() << " GPU tests passed.\n";
    return (passed == static_cast<int>(cases.size())) ? 0 : 1;
}
