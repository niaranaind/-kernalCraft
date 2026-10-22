//
// MiniPTX Micro-Benchmark: MiniPTX vs Hand-Written CUDA SAXPY
//
// Pure CUDA Driver API implementation (cuEvent*, cuLaunchKernel, cuModuleLoadDataEx).
// Target: NVIDIA Tesla V100 (sm_70) on DGX.
//

#include <cuda.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

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
            return 1;                                                       \
        }                                                                   \
    } while (0)

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

static int count_ptx_instructions(const std::string& ptx) {
    std::stringstream ss(ptx);
    std::string line;
    int count = 0;
    bool in_entry = false;

    while (std::getline(ss, line)) {
        size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos) continue;
        std::string trimmed = line.substr(first);

        if (trimmed.find(".entry ") != std::string::npos) {
            in_entry = true;
            continue;
        }
        if (!in_entry) continue;

        // Skip comments, directives, labels, closing brace
        if (trimmed[0] == '/' || trimmed[0] == ';' || trimmed[0] == '.' || trimmed[0] == '}' || trimmed[0] == '{') {
            continue;
        }
        if (trimmed.back() == ':') {
            continue;
        }
        // An instruction ends with ';'
        if (trimmed.find(';') != std::string::npos) {
            count++;
        }
    }
    return count;
}

struct BenchmarkStats {
    float median_ms = 0.0f;
    float min_ms = 0.0f;
    float max_ms = 0.0f;
    float mean_ms = 0.0f;
    float stddev_ms = 0.0f;
    float bandwidth_gb_s = 0.0f;
};

static BenchmarkStats compute_stats(std::vector<float> times, size_t n_elements) {
    BenchmarkStats s;
    if (times.empty()) return s;

    std::sort(times.begin(), times.end());
    size_t sz = times.size();
    s.min_ms = times.front();
    s.max_ms = times.back();

    if (sz % 2 == 1) {
        s.median_ms = times[sz / 2];
    } else {
        s.median_ms = (times[sz / 2 - 1] + times[sz / 2]) / 2.0f;
    }

    float sum = std::accumulate(times.begin(), times.end(), 0.0f);
    s.mean_ms = sum / static_cast<float>(sz);

    float var_sum = 0.0f;
    for (float t : times) {
        var_sum += (t - s.mean_ms) * (t - s.mean_ms);
    }
    s.stddev_ms = std::sqrt(var_sum / static_cast<float>(sz));

    // Effective bandwidth: 3 memory accesses per element (read x, read y, write y)
    // Traffic = 3 * N * sizeof(float) bytes
    double total_bytes = 3.0 * static_cast<double>(n_elements) * sizeof(float);
    double median_seconds = static_cast<double>(s.median_ms) / 1000.0;
    s.bandwidth_gb_s = static_cast<float>((total_bytes / median_seconds) / 1e9);

    return s;
}

int main(int argc, char** argv) {
    std::string miniptx_ptx_path = (argc > 1) ? argv[1] : "build_dgx/saxpy.ptx";
    std::string cuda_ptx_path    = (argc > 2) ? argv[2] : "build_dgx/cuda_saxpy.ptx";

    std::cout << "=================================================================\n";
    std::cout << " MiniPTX Micro-Benchmark: MiniPTX vs. Hand-Written CUDA SAXPY    \n";
    std::cout << " Target: NVIDIA Tesla V100-SXM2 (sm_70) via CUDA Driver API      \n";
    std::cout << "=================================================================\n\n";

    // 1. Read PTX sources and count instructions
    std::string miniptx_ptx = read_file(miniptx_ptx_path);
    std::string cuda_ptx    = read_file(cuda_ptx_path);
    if (miniptx_ptx.empty() || cuda_ptx.empty()) {
        std::cerr << "error: failed to load one or both PTX files.\n";
        return 1;
    }

    int miniptx_instr_count = count_ptx_instructions(miniptx_ptx);
    int cuda_instr_count    = count_ptx_instructions(cuda_ptx);

    std::cout << "PTX Instruction Counts (kernel body):\n";
    std::cout << "  - MiniPTX:         " << miniptx_instr_count << " instructions\n";
    std::cout << "  - Hand-written CUDA (nvcc -O3): " << cuda_instr_count << " instructions\n\n";

    // 2. Initialize CUDA Driver
    CUDA_CHECK(cuInit(0));

    int device_count = 0;
    CUDA_CHECK(cuDeviceGetCount(&device_count));
    if (device_count == 0) {
        std::cerr << "error: no CUDA devices found.\n";
        return 1;
    }

    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    char dev_name[256];
    CUDA_CHECK(cuDeviceGetName(dev_name, sizeof(dev_name), device));
    std::cout << "Selected GPU Device 0: " << dev_name << "\n";

    // Check available memory
    CUcontext context;
    CUDA_CHECK(cuCtxCreate(&context, 0, device));

    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cuMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU Memory: " << (free_mem / (1024 * 1024)) << " MiB free / "
              << (total_mem / (1024 * 1024)) << " MiB total\n";

    // Problem size: N = 2^24 = 16,777,216 floats
    const int N = 1 << 24;
    const size_t bytes = static_cast<size_t>(N) * sizeof(float);
    const size_t total_footprint_mb = (2 * bytes) / (1024 * 1024);
    const size_t traffic_mb = (3 * bytes) / (1024 * 1024);

    std::cout << "Workload Parameters:\n";
    std::cout << "  - Array size N: " << N << " elements (2^24 floats)\n";
    std::cout << "  - Footprint:   ~" << total_footprint_mb << " MiB (vectors x and y in V100 HBM2)\n";
    std::cout << "  - Data Traffic: ~" << traffic_mb << " MiB per iteration (read x + read y + write y)\n";
    std::cout << "  - Note: Tesla V100 L2 cache is 6 MiB (workload completely spills to DRAM)\n\n";

    if (free_mem < 2 * bytes + 100 * 1024 * 1024) {
        std::cerr << "error: not enough free memory on GPU (" << (free_mem / (1024 * 1024)) << " MiB free).\n";
        CUDA_CHECK(cuCtxDestroy(context));
        return 1;
    }

    // 3. Load both modules via Driver API
    char err_log_a[2048] = {0};
    char err_log_b[2048] = {0};
    CUjit_option jit_options[] = {
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
    };

    void* jit_opt_vals_a[] = { err_log_a, reinterpret_cast<void*>(sizeof(err_log_a)) };
    void* jit_opt_vals_b[] = { err_log_b, reinterpret_cast<void*>(sizeof(err_log_b)) };

    CUmodule mod_a, mod_b;
    CUresult res_a = cuModuleLoadDataEx(&mod_a, miniptx_ptx.c_str(), 2, jit_options, jit_opt_vals_a);
    if (res_a != CUDA_SUCCESS) {
        std::cerr << "error: failed to load MiniPTX module:\n" << err_log_a << "\n";
        return 1;
    }

    CUresult res_b = cuModuleLoadDataEx(&mod_b, cuda_ptx.c_str(), 2, jit_options, jit_opt_vals_b);
    if (res_b != CUDA_SUCCESS) {
        std::cerr << "error: failed to load CUDA module:\n" << err_log_b << "\n";
        return 1;
    }

    CUfunction fn_a, fn_b;
    CUDA_CHECK(cuModuleGetFunction(&fn_a, mod_a, "saxpy"));
    CUDA_CHECK(cuModuleGetFunction(&fn_b, mod_b, "saxpy"));

    // 4. Allocate GPU memory & initialize host test data
    std::vector<float> h_x(N), h_y_init(N), h_ref(N), h_y_verify(N);
    const float a_val = 2.5f;

    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.0f + static_cast<float>(i % 100) * 0.01f;
        h_y_init[i] = 2.0f + static_cast<float>(i % 50) * 0.05f;
        h_ref[i] = a_val * h_x[i] + h_y_init[i];
    }

    CUdeviceptr d_x, d_y;
    CUDA_CHECK(cuMemAlloc(&d_x, bytes));
    CUDA_CHECK(cuMemAlloc(&d_y, bytes));

    CUDA_CHECK(cuMemcpyHtoD(d_x, h_x.data(), bytes));

    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    float a_param = a_val;
    int n_param = N;

    void* kernel_args[] = { &a_param, &d_x, &d_y, &n_param };

    // 5. Verification Check (Run 1 iteration of each and verify vs CPU)
    std::cout << "--- Step 1: Correctness Verification ---\n";
    // Verify MiniPTX
    CUDA_CHECK(cuMemcpyHtoD(d_y, h_y_init.data(), bytes));
    CUDA_CHECK(cuLaunchKernel(fn_a, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, kernel_args, nullptr));
    CUDA_CHECK(cuCtxSynchronize());
    CUDA_CHECK(cuMemcpyDtoH(h_y_verify.data(), d_y, bytes));
    for (int i = 0; i < N; ++i) {
        if (std::abs(h_y_verify[i] - h_ref[i]) > 1e-4f) {
            std::cerr << "Verification failed for MiniPTX at index " << i
                      << ": got " << h_y_verify[i] << ", expected " << h_ref[i] << "\n";
            return 1;
        }
    }
    std::cout << "  MiniPTX kernel output: [PASS] Correct against CPU reference.\n";

    // Verify Hand-written CUDA
    CUDA_CHECK(cuMemcpyHtoD(d_y, h_y_init.data(), bytes));
    CUDA_CHECK(cuLaunchKernel(fn_b, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, kernel_args, nullptr));
    CUDA_CHECK(cuCtxSynchronize());
    CUDA_CHECK(cuMemcpyDtoH(h_y_verify.data(), d_y, bytes));
    for (int i = 0; i < N; ++i) {
        if (std::abs(h_y_verify[i] - h_ref[i]) > 1e-4f) {
            std::cerr << "Verification failed for CUDA kernel at index " << i
                      << ": got " << h_y_verify[i] << ", expected " << h_ref[i] << "\n";
            return 1;
        }
    }
    std::cout << "  CUDA kernel output:    [PASS] Correct against CPU reference.\n\n";

    // 6. Warm-up Iterations (Alternating A/B)
    std::cout << "--- Step 2: Warm-up Runs (10 iterations each, alternating A/B) ---\n";
    for (int iter = 0; iter < 10; ++iter) {
        CUDA_CHECK(cuLaunchKernel(fn_a, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, kernel_args, nullptr));
        CUDA_CHECK(cuLaunchKernel(fn_b, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, kernel_args, nullptr));
    }
    CUDA_CHECK(cuCtxSynchronize());
    std::cout << "  Warm-up complete.\n\n";

    // 7. Benchmark Measurement (Alternating A/B with cuEvent*)
    const int num_timed_runs = 50;
    std::cout << "--- Step 3: Timed Benchmark Runs (" << num_timed_runs << " iterations each, alternating A/B) ---\n";

    CUevent start_a, stop_a;
    CUevent start_b, stop_b;
    CUDA_CHECK(cuEventCreate(&start_a, CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventCreate(&stop_a,  CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventCreate(&start_b, CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventCreate(&stop_b,  CU_EVENT_DEFAULT));

    std::vector<float> times_miniptx;
    std::vector<float> times_cuda;
    times_miniptx.reserve(num_timed_runs);
    times_cuda.reserve(num_timed_runs);

    for (int iter = 0; iter < num_timed_runs; ++iter) {
        // Run A (MiniPTX)
        CUDA_CHECK(cuEventRecord(start_a, 0));
        CUDA_CHECK(cuLaunchKernel(fn_a, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, kernel_args, nullptr));
        CUDA_CHECK(cuEventRecord(stop_a, 0));
        CUDA_CHECK(cuEventSynchronize(stop_a));
        float ms_a = 0.0f;
        CUDA_CHECK(cuEventElapsedTime(&ms_a, start_a, stop_a));
        times_miniptx.push_back(ms_a);

        // Run B (Hand-written CUDA)
        CUDA_CHECK(cuEventRecord(start_b, 0));
        CUDA_CHECK(cuLaunchKernel(fn_b, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, kernel_args, nullptr));
        CUDA_CHECK(cuEventRecord(stop_b, 0));
        CUDA_CHECK(cuEventSynchronize(stop_b));
        float ms_b = 0.0f;
        CUDA_CHECK(cuEventElapsedTime(&ms_b, start_b, stop_b));
        times_cuda.push_back(ms_b);
    }

    // 8. Compute Statistics
    BenchmarkStats stats_miniptx = compute_stats(times_miniptx, N);
    BenchmarkStats stats_cuda    = compute_stats(times_cuda, N);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=================================================================\n";
    std::cout << " Benchmark Results (N = 2^24 floats, 50 alternating iterations)  \n";
    std::cout << "=================================================================\n\n";

    std::cout << "Kernel Execution Times (ms):\n";
    std::cout << "  ------------------------------------------------------------------------------------------\n";
    std::cout << "  Implementation          | Median (ms) | Min (ms)  | Max (ms)  | StdDev (ms) | BW (GB/s)   \n";
    std::cout << "  ------------------------------------------------------------------------------------------\n";
    std::cout << "  MiniPTX (emitted PTX)   | " << std::setw(11) << stats_miniptx.median_ms
              << " | " << std::setw(9) << stats_miniptx.min_ms
              << " | " << std::setw(9) << stats_miniptx.max_ms
              << " | " << std::setw(11) << stats_miniptx.stddev_ms
              << " | " << std::setw(9) << stats_miniptx.bandwidth_gb_s << " \n";
    std::cout << "  CUDA (nvcc -O3 PTX)     | " << std::setw(11) << stats_cuda.median_ms
              << " | " << std::setw(9) << stats_cuda.min_ms
              << " | " << std::setw(9) << stats_cuda.max_ms
              << " | " << std::setw(11) << stats_cuda.stddev_ms
              << " | " << std::setw(9) << stats_cuda.bandwidth_gb_s << " \n";
    std::cout << "  ------------------------------------------------------------------------------------------\n\n";

    float ratio = stats_miniptx.median_ms / stats_cuda.median_ms;
    std::cout << "Relative Runtime (MiniPTX / CUDA nvcc -O3): " << std::setprecision(3) << ratio << "x\n";
    if (ratio <= 1.05f) {
        std::cout << "  => MiniPTX matches nvcc -O3 performance within 5%!\n";
    }

    std::cout << "\nEnvironment Note:\n";
    std::cout << "  - Measurements taken on a shared DGX system (" << dev_name << ").\n";
    std::cout << "  - Alternating A/B testing minimizes power-state and dynamic clock skew.\n";
    std::cout << "  - Bandwidth formula: (3 * N * 4 bytes) / median_runtime_seconds.\n";

    // Cleanup
    CUDA_CHECK(cuEventDestroy(start_a));
    CUDA_CHECK(cuEventDestroy(stop_a));
    CUDA_CHECK(cuEventDestroy(start_b));
    CUDA_CHECK(cuEventDestroy(stop_b));
    CUDA_CHECK(cuMemFree(d_x));
    CUDA_CHECK(cuMemFree(d_y));
    CUDA_CHECK(cuModuleUnload(mod_a));
    CUDA_CHECK(cuModuleUnload(mod_b));
    CUDA_CHECK(cuCtxDestroy(context));

    return 0;
}
