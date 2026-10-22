//
// MiniPTX Micro-Benchmark: MiniPTX vs Hand-Written CUDA
//
// Pure CUDA Driver API implementation (cuEvent*, cuLaunchKernel, cuModuleLoadDataEx).
// Target: NVIDIA Tesla V100-SXM2 (sm_70) on DGX.
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

#define CUDA_CHECK_RET(expr, ret_val)                                       \
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
            return (ret_val);                                               \
        }                                                                   \
    } while (0)

#define CUDA_CHECK(expr) CUDA_CHECK_RET(expr, false)
#define CUDA_CHECK_MAIN(expr) CUDA_CHECK_RET(expr, 1)

static std::string read_file(const std::string& path, bool required = true) {
    std::ifstream in(path);
    if (!in) {
        if (required) {
            std::cerr << "error: cannot open file '" << path << "'\n";
        }
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

        if (trimmed[0] == '/' || trimmed[0] == ';' || trimmed[0] == '.' || trimmed[0] == '}' || trimmed[0] == '{') {
            continue;
        }
        if (trimmed.back() == ':') {
            continue;
        }
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
};

static BenchmarkStats compute_stats(std::vector<float> times) {
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

    return s;
}

static void print_verdict(const std::string& label_a, const std::string& label_b, float ratio) {
    std::cout << "Relative Runtime (" << label_a << " / " << label_b << "): "
              << std::fixed << std::setprecision(3) << ratio << "x\n";
    float diff_pct = std::abs(ratio - 1.0f) * 100.0f;
    if (diff_pct <= 5.0f) {
        std::cout << "  => Verdict: " << label_a << " performance matches " << label_b
                  << " within 5% (measured diff: " << std::setprecision(2) << diff_pct << "%).\n";
    } else if (ratio < 1.0f) {
        std::cout << "  => Verdict: " << label_a << " is " << std::setprecision(2)
                  << (1.0f - ratio) * 100.0f << "% faster than " << label_b
                  << " (" << std::setprecision(3) << ratio << "x).\n";
    } else {
        std::cout << "  => Verdict: " << label_a << " is " << std::setprecision(2)
                  << (ratio - 1.0f) * 100.0f << "% slower than " << label_b
                  << " (" << std::setprecision(3) << ratio << "x).\n";
    }
}

// ---------------------------------------------------------------------------
// Benchmark 1: SAXPY (Memory-Bound)
// ---------------------------------------------------------------------------
static bool run_saxpy_benchmark(const std::string& miniptx_ptx, const std::string& cuda_ptx) {
    std::cout << "=================================================================\n";
    std::cout << " Benchmark 1: SAXPY (Memory-Bound, N = 2^24 elements)            \n";
    std::cout << "=================================================================\n\n";

    int miniptx_instr_count = count_ptx_instructions(miniptx_ptx);
    int cuda_instr_count    = count_ptx_instructions(cuda_ptx);

    std::cout << "PTX Instruction Counts (kernel body):\n";
    std::cout << "  - MiniPTX:                    " << miniptx_instr_count << " instructions\n";
    std::cout << "  - Hand-written CUDA (nvcc -O3): " << cuda_instr_count << " instructions\n\n";

    const int N = 1 << 24;
    const size_t bytes = static_cast<size_t>(N) * sizeof(float);

    char err_log_a[2048] = {0};
    char err_log_b[2048] = {0};
    CUjit_option jit_options[] = {
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
    };
    void* jit_opt_vals_a[] = { err_log_a, reinterpret_cast<void*>(sizeof(err_log_a)) };
    void* jit_opt_vals_b[] = { err_log_b, reinterpret_cast<void*>(sizeof(err_log_b)) };

    CUmodule mod_a, mod_b;
    if (cuModuleLoadDataEx(&mod_a, miniptx_ptx.c_str(), 2, jit_options, jit_opt_vals_a) != CUDA_SUCCESS) {
        std::cerr << "error: failed to load MiniPTX SAXPY module:\n" << err_log_a << "\n";
        return false;
    }
    if (cuModuleLoadDataEx(&mod_b, cuda_ptx.c_str(), 2, jit_options, jit_opt_vals_b) != CUDA_SUCCESS) {
        std::cerr << "error: failed to load CUDA SAXPY module:\n" << err_log_b << "\n";
        cuModuleUnload(mod_a);
        return false;
    }

    CUfunction fn_a, fn_b;
    CUDA_CHECK(cuModuleGetFunction(&fn_a, mod_a, "saxpy"));
    CUDA_CHECK(cuModuleGetFunction(&fn_b, mod_b, "saxpy"));

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

    // 1. Correctness Verification
    std::cout << "--- Correctness Verification (on fresh buffer) ---\n";
    CUDA_CHECK(cuMemcpyHtoD(d_y, h_y_init.data(), bytes));
    CUDA_CHECK(cuLaunchKernel(fn_a, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, kernel_args, nullptr));
    CUDA_CHECK(cuCtxSynchronize());
    CUDA_CHECK(cuMemcpyDtoH(h_y_verify.data(), d_y, bytes));
    for (int i = 0; i < N; ++i) {
        if (std::abs(h_y_verify[i] - h_ref[i]) > 1e-4f) {
            std::cerr << "Verification failed for MiniPTX SAXPY at index " << i << "\n";
            cuMemFree(d_x);
            cuMemFree(d_y);
            cuModuleUnload(mod_a);
            cuModuleUnload(mod_b);
            return false;
        }
    }
    std::cout << "  MiniPTX SAXPY: [PASS] Correct against CPU reference.\n";

    CUDA_CHECK(cuMemcpyHtoD(d_y, h_y_init.data(), bytes));
    CUDA_CHECK(cuLaunchKernel(fn_b, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, kernel_args, nullptr));
    CUDA_CHECK(cuCtxSynchronize());
    CUDA_CHECK(cuMemcpyDtoH(h_y_verify.data(), d_y, bytes));
    for (int i = 0; i < N; ++i) {
        if (std::abs(h_y_verify[i] - h_ref[i]) > 1e-4f) {
            std::cerr << "Verification failed for CUDA SAXPY at index " << i << "\n";
            cuMemFree(d_x);
            cuMemFree(d_y);
            cuModuleUnload(mod_a);
            cuModuleUnload(mod_b);
            return false;
        }
    }
    std::cout << "  CUDA SAXPY:    [PASS] Correct against CPU reference.\n\n";

    // 2. Warm-up (10 alternating iterations)
    std::cout << "--- Warm-up (10 alternating iterations) ---\n";
    for (int i = 0; i < 10; ++i) {
        CUDA_CHECK(cuLaunchKernel(fn_a, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, kernel_args, nullptr));
        CUDA_CHECK(cuLaunchKernel(fn_b, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, kernel_args, nullptr));
    }
    CUDA_CHECK(cuCtxSynchronize());

    // 3. Timed Runs (50 alternating iterations with cuEvent*)
    const int num_runs = 50;
    std::cout << "--- Timed Runs (" << num_runs << " alternating iterations) ---\n";
    CUevent start_a, stop_a, start_b, stop_b;
    CUDA_CHECK(cuEventCreate(&start_a, CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventCreate(&stop_a,  CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventCreate(&start_b, CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventCreate(&stop_b,  CU_EVENT_DEFAULT));

    std::vector<float> times_a, times_b;
    times_a.reserve(num_runs);
    times_b.reserve(num_runs);

    for (int i = 0; i < num_runs; ++i) {
        CUDA_CHECK(cuEventRecord(start_a, 0));
        CUDA_CHECK(cuLaunchKernel(fn_a, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, kernel_args, nullptr));
        CUDA_CHECK(cuEventRecord(stop_a, 0));
        CUDA_CHECK(cuEventSynchronize(stop_a));
        float ms_a = 0.0f;
        CUDA_CHECK(cuEventElapsedTime(&ms_a, start_a, stop_a));
        times_a.push_back(ms_a);

        CUDA_CHECK(cuEventRecord(start_b, 0));
        CUDA_CHECK(cuLaunchKernel(fn_b, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, kernel_args, nullptr));
        CUDA_CHECK(cuEventRecord(stop_b, 0));
        CUDA_CHECK(cuEventSynchronize(stop_b));
        float ms_b = 0.0f;
        CUDA_CHECK(cuEventElapsedTime(&ms_b, start_b, stop_b));
        times_b.push_back(ms_b);
    }

    BenchmarkStats stats_a = compute_stats(times_a);
    BenchmarkStats stats_b = compute_stats(times_b);

    double traffic_bytes = 3.0 * static_cast<double>(N) * sizeof(float);
    float bw_a = static_cast<float>((traffic_bytes / (static_cast<double>(stats_a.median_ms) / 1000.0)) / 1e9);
    float bw_b = static_cast<float>((traffic_bytes / (static_cast<double>(stats_b.median_ms) / 1000.0)) / 1e9);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "SAXPY Execution Times (ms):\n";
    std::cout << "  ------------------------------------------------------------------------------------------\n";
    std::cout << "  Implementation          | Median (ms) | Min (ms)  | Max (ms)  | StdDev (ms) | BW (GB/s)   \n";
    std::cout << "  ------------------------------------------------------------------------------------------\n";
    std::cout << "  MiniPTX (emitted PTX)   | " << std::setw(11) << stats_a.median_ms
              << " | " << std::setw(9) << stats_a.min_ms
              << " | " << std::setw(9) << stats_a.max_ms
              << " | " << std::setw(11) << stats_a.stddev_ms
              << " | " << std::setw(9) << bw_a << " \n";
    std::cout << "  CUDA (nvcc -O3 PTX)     | " << std::setw(11) << stats_b.median_ms
              << " | " << std::setw(9) << stats_b.min_ms
              << " | " << std::setw(9) << stats_b.max_ms
              << " | " << std::setw(11) << stats_b.stddev_ms
              << " | " << std::setw(9) << bw_b << " \n";
    std::cout << "  ------------------------------------------------------------------------------------------\n\n";

    float ratio = stats_a.median_ms / stats_b.median_ms;
    print_verdict("MiniPTX", "CUDA nvcc -O3", ratio);

    CUDA_CHECK(cuEventDestroy(start_a));
    CUDA_CHECK(cuEventDestroy(stop_a));
    CUDA_CHECK(cuEventDestroy(start_b));
    CUDA_CHECK(cuEventDestroy(stop_b));
    CUDA_CHECK(cuMemFree(d_x));
    CUDA_CHECK(cuMemFree(d_y));
    CUDA_CHECK(cuModuleUnload(mod_a));
    CUDA_CHECK(cuModuleUnload(mod_b));
    std::cout << "\n";
    return true;
}

// ---------------------------------------------------------------------------
// Benchmark 2: Compute-Heavy (Inner Loop, N = 2^24 elements, 64 iters)
// ---------------------------------------------------------------------------
static bool run_compute_benchmark(const std::string& miniptx_ptx,
                                  const std::string& cuda_ptx,
                                  const std::string& cuda_nofma_ptx = "") {
    bool has_nofma = !cuda_nofma_ptx.empty();

    std::cout << "=================================================================\n";
    std::cout << " Benchmark 2: Compute-Heavy (N = 2^24 elements, 64-iter inner loop)\n";
    std::cout << "=================================================================\n\n";

    int miniptx_instr_count    = count_ptx_instructions(miniptx_ptx);
    int cuda_instr_count       = count_ptx_instructions(cuda_ptx);
    int cuda_nofma_instr_count = has_nofma ? count_ptx_instructions(cuda_nofma_ptx) : 0;

    std::cout << "PTX Instruction Counts (kernel body):\n";
    std::cout << "  - MiniPTX:                           " << miniptx_instr_count << " instructions\n";
    std::cout << "  - Hand-written CUDA (nvcc -O3):        " << cuda_instr_count << " instructions\n";
    if (has_nofma) {
        std::cout << "  - CUDA no-FMA (nvcc -O3 --fmad=false): " << cuda_nofma_instr_count << " instructions\n";
    }
    std::cout << "\n";

    const int N = 1 << 24;
    const int iters = 64;
    const size_t bytes = static_cast<size_t>(N) * sizeof(float);

    char err_log_a[2048] = {0};
    char err_log_b[2048] = {0};
    char err_log_c[2048] = {0};
    CUjit_option jit_options[] = {
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
    };
    void* jit_opt_vals_a[] = { err_log_a, reinterpret_cast<void*>(sizeof(err_log_a)) };
    void* jit_opt_vals_b[] = { err_log_b, reinterpret_cast<void*>(sizeof(err_log_b)) };
    void* jit_opt_vals_c[] = { err_log_c, reinterpret_cast<void*>(sizeof(err_log_c)) };

    CUmodule mod_a = nullptr, mod_b = nullptr, mod_c = nullptr;
    if (cuModuleLoadDataEx(&mod_a, miniptx_ptx.c_str(), 2, jit_options, jit_opt_vals_a) != CUDA_SUCCESS) {
        std::cerr << "error: failed to load MiniPTX compute_heavy module:\n" << err_log_a << "\n";
        return false;
    }
    if (cuModuleLoadDataEx(&mod_b, cuda_ptx.c_str(), 2, jit_options, jit_opt_vals_b) != CUDA_SUCCESS) {
        std::cerr << "error: failed to load CUDA compute_heavy module:\n" << err_log_b << "\n";
        cuModuleUnload(mod_a);
        return false;
    }
    if (has_nofma) {
        if (cuModuleLoadDataEx(&mod_c, cuda_nofma_ptx.c_str(), 2, jit_options, jit_opt_vals_c) != CUDA_SUCCESS) {
            std::cerr << "error: failed to load CUDA no-FMA compute_heavy module:\n" << err_log_c << "\n";
            cuModuleUnload(mod_a);
            cuModuleUnload(mod_b);
            return false;
        }
    }

    CUfunction fn_a, fn_b, fn_c = nullptr;
    CUDA_CHECK(cuModuleGetFunction(&fn_a, mod_a, "compute_heavy"));
    CUDA_CHECK(cuModuleGetFunction(&fn_b, mod_b, "compute_heavy"));
    if (has_nofma) {
        CUDA_CHECK(cuModuleGetFunction(&fn_c, mod_c, "compute_heavy"));
    }

    std::vector<float> h_x(N), h_ref(N), h_y_verify(N);
    const float a_val = 1.0001f;
    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.0f + static_cast<float>(i % 100) * 0.01f;
        float v = h_x[i];
        for (int j = 0; j < iters; ++j) {
            v = v * a_val + 0.5f;
        }
        h_ref[i] = v;
    }

    CUdeviceptr d_x, d_y;
    CUDA_CHECK(cuMemAlloc(&d_x, bytes));
    CUDA_CHECK(cuMemAlloc(&d_y, bytes));
    CUDA_CHECK(cuMemcpyHtoD(d_x, h_x.data(), bytes));

    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    float a_param = a_val;
    int n_param = N;
    int iters_param = iters;
    void* kernel_args[] = { &d_x, &d_y, &a_param, &n_param, &iters_param };

    // 1. Correctness Verification
    std::cout << "--- Correctness Verification (on fresh buffer) ---\n";
    CUDA_CHECK(cuLaunchKernel(fn_a, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, kernel_args, nullptr));
    CUDA_CHECK(cuCtxSynchronize());
    CUDA_CHECK(cuMemcpyDtoH(h_y_verify.data(), d_y, bytes));
    for (int i = 0; i < N; ++i) {
        if (std::abs(h_y_verify[i] - h_ref[i]) > 1e-2f * std::abs(h_ref[i])) {
            std::cerr << "Verification failed for MiniPTX compute_heavy at index " << i << "\n";
            cuMemFree(d_x);
            cuMemFree(d_y);
            cuModuleUnload(mod_a);
            cuModuleUnload(mod_b);
            if (mod_c) cuModuleUnload(mod_c);
            return false;
        }
    }
    std::cout << "  MiniPTX compute_heavy:     [PASS] Correct against CPU reference.\n";

    CUDA_CHECK(cuLaunchKernel(fn_b, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, kernel_args, nullptr));
    CUDA_CHECK(cuCtxSynchronize());
    CUDA_CHECK(cuMemcpyDtoH(h_y_verify.data(), d_y, bytes));
    for (int i = 0; i < N; ++i) {
        if (std::abs(h_y_verify[i] - h_ref[i]) > 1e-2f * std::abs(h_ref[i])) {
            std::cerr << "Verification failed for CUDA compute_heavy at index " << i << "\n";
            cuMemFree(d_x);
            cuMemFree(d_y);
            cuModuleUnload(mod_a);
            cuModuleUnload(mod_b);
            if (mod_c) cuModuleUnload(mod_c);
            return false;
        }
    }
    std::cout << "  CUDA compute_heavy (O3):   [PASS] Correct against CPU reference.\n";

    if (has_nofma) {
        CUDA_CHECK(cuLaunchKernel(fn_c, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, kernel_args, nullptr));
        CUDA_CHECK(cuCtxSynchronize());
        CUDA_CHECK(cuMemcpyDtoH(h_y_verify.data(), d_y, bytes));
        for (int i = 0; i < N; ++i) {
            if (std::abs(h_y_verify[i] - h_ref[i]) > 1e-2f * std::abs(h_ref[i])) {
                std::cerr << "Verification failed for CUDA no-FMA compute_heavy at index " << i << "\n";
                cuMemFree(d_x);
                cuMemFree(d_y);
                cuModuleUnload(mod_a);
                cuModuleUnload(mod_b);
                cuModuleUnload(mod_c);
                return false;
            }
        }
        std::cout << "  CUDA no-FMA compute_heavy: [PASS] Correct against CPU reference.\n";
    }
    std::cout << "\n";

    // 2. Warm-up (10 alternating iterations)
    std::cout << "--- Warm-up (10 alternating iterations) ---\n";
    for (int i = 0; i < 10; ++i) {
        CUDA_CHECK(cuLaunchKernel(fn_a, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, kernel_args, nullptr));
        CUDA_CHECK(cuLaunchKernel(fn_b, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, kernel_args, nullptr));
        if (has_nofma) {
            CUDA_CHECK(cuLaunchKernel(fn_c, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, kernel_args, nullptr));
        }
    }
    CUDA_CHECK(cuCtxSynchronize());

    // 3. Timed Runs (50 alternating iterations with cuEvent*)
    const int num_runs = 50;
    std::cout << "--- Timed Runs (" << num_runs << " alternating iterations) ---\n";
    CUevent start_a, stop_a, start_b, stop_b, start_c, stop_c;
    CUDA_CHECK(cuEventCreate(&start_a, CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventCreate(&stop_a,  CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventCreate(&start_b, CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventCreate(&stop_b,  CU_EVENT_DEFAULT));
    if (has_nofma) {
        CUDA_CHECK(cuEventCreate(&start_c, CU_EVENT_DEFAULT));
        CUDA_CHECK(cuEventCreate(&stop_c,  CU_EVENT_DEFAULT));
    }

    std::vector<float> times_a, times_b, times_c;
    times_a.reserve(num_runs);
    times_b.reserve(num_runs);
    if (has_nofma) times_c.reserve(num_runs);

    for (int i = 0; i < num_runs; ++i) {
        CUDA_CHECK(cuEventRecord(start_a, 0));
        CUDA_CHECK(cuLaunchKernel(fn_a, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, kernel_args, nullptr));
        CUDA_CHECK(cuEventRecord(stop_a, 0));
        CUDA_CHECK(cuEventSynchronize(stop_a));
        float ms_a = 0.0f;
        CUDA_CHECK(cuEventElapsedTime(&ms_a, start_a, stop_a));
        times_a.push_back(ms_a);

        CUDA_CHECK(cuEventRecord(start_b, 0));
        CUDA_CHECK(cuLaunchKernel(fn_b, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, kernel_args, nullptr));
        CUDA_CHECK(cuEventRecord(stop_b, 0));
        CUDA_CHECK(cuEventSynchronize(stop_b));
        float ms_b = 0.0f;
        CUDA_CHECK(cuEventElapsedTime(&ms_b, start_b, stop_b));
        times_b.push_back(ms_b);

        if (has_nofma) {
            CUDA_CHECK(cuEventRecord(start_c, 0));
            CUDA_CHECK(cuLaunchKernel(fn_c, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, kernel_args, nullptr));
            CUDA_CHECK(cuEventRecord(stop_c, 0));
            CUDA_CHECK(cuEventSynchronize(stop_c));
            float ms_c = 0.0f;
            CUDA_CHECK(cuEventElapsedTime(&ms_c, start_c, stop_c));
            times_c.push_back(ms_c);
        }
    }

    BenchmarkStats stats_a = compute_stats(times_a);
    BenchmarkStats stats_b = compute_stats(times_b);

    double total_flops = static_cast<double>(N) * static_cast<double>(iters) * 2.0;
    float gflops_a = static_cast<float>((total_flops / (static_cast<double>(stats_a.median_ms) / 1000.0)) / 1e9);
    float gflops_b = static_cast<float>((total_flops / (static_cast<double>(stats_b.median_ms) / 1000.0)) / 1e9);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Compute-Heavy Execution Times (ms):\n";
    std::cout << "  -------------------------------------------------------------------------------------------------------\n";
    std::cout << "  Implementation                      | Median (ms) | Min (ms)  | Max (ms)  | StdDev (ms) | GFLOPS      \n";
    std::cout << "  -------------------------------------------------------------------------------------------------------\n";
    std::cout << "  MiniPTX (emitted PTX)               | " << std::setw(11) << stats_a.median_ms
              << " | " << std::setw(9) << stats_a.min_ms
              << " | " << std::setw(9) << stats_a.max_ms
              << " | " << std::setw(11) << stats_a.stddev_ms
              << " | " << std::setw(9) << gflops_a << " \n";
    std::cout << "  CUDA (nvcc -O3 PTX)                 | " << std::setw(11) << stats_b.median_ms
              << " | " << std::setw(9) << stats_b.min_ms
              << " | " << std::setw(9) << stats_b.max_ms
              << " | " << std::setw(11) << stats_b.stddev_ms
              << " | " << std::setw(9) << gflops_b << " \n";

    if (has_nofma) {
        BenchmarkStats stats_c = compute_stats(times_c);
        float gflops_c = static_cast<float>((total_flops / (static_cast<double>(stats_c.median_ms) / 1000.0)) / 1e9);
        std::cout << "  CUDA no-FMA (nvcc -O3 --fmad=false) | " << std::setw(11) << stats_c.median_ms
                  << " | " << std::setw(9) << stats_c.min_ms
                  << " | " << std::setw(9) << stats_c.max_ms
                  << " | " << std::setw(11) << stats_c.stddev_ms
                  << " | " << std::setw(9) << gflops_c << " \n";
    }
    std::cout << "  -------------------------------------------------------------------------------------------------------\n\n";

    float ratio_ab = stats_a.median_ms / stats_b.median_ms;
    print_verdict("MiniPTX", "CUDA nvcc -O3", ratio_ab);

    if (has_nofma) {
        BenchmarkStats stats_c = compute_stats(times_c);
        float ratio_ac = stats_a.median_ms / stats_c.median_ms;
        print_verdict("MiniPTX", "CUDA no-FMA (--fmad=false)", ratio_ac);

        float ratio_bc = stats_b.median_ms / stats_c.median_ms;
        std::cout << "FMA Diagnostic: CUDA (nvcc -O3 with FMA) vs CUDA no-FMA (--fmad=false):\n";
        std::cout << "  Ratio: " << std::fixed << std::setprecision(3) << ratio_bc << "x (";
        if (ratio_bc < 1.0f) {
            std::cout << std::setprecision(2) << (1.0f - ratio_bc) * 100.0f << "% speedup from hardware FMA fusion)\n";
        } else {
            std::cout << std::setprecision(2) << (ratio_bc - 1.0f) * 100.0f << "% slowdown)\n";
        }
    }

    CUDA_CHECK(cuEventDestroy(start_a));
    CUDA_CHECK(cuEventDestroy(stop_a));
    CUDA_CHECK(cuEventDestroy(start_b));
    CUDA_CHECK(cuEventDestroy(stop_b));
    if (has_nofma) {
        CUDA_CHECK(cuEventDestroy(start_c));
        CUDA_CHECK(cuEventDestroy(stop_c));
    }
    CUDA_CHECK(cuMemFree(d_x));
    CUDA_CHECK(cuMemFree(d_y));
    CUDA_CHECK(cuModuleUnload(mod_a));
    CUDA_CHECK(cuModuleUnload(mod_b));
    if (mod_c) CUDA_CHECK(cuModuleUnload(mod_c));
    std::cout << "\n";
    return true;
}

int main(int argc, char** argv) {
    std::string build_dir = "build_dgx";
    std::string saxpy_miniptx_path;
    std::string saxpy_cuda_path;
    std::string compute_miniptx_path;
    std::string compute_cuda_path;
    std::string compute_nofma_path;

    if (argc == 2) {
        build_dir = argv[1];
        saxpy_miniptx_path   = build_dir + "/miniptx_saxpy.ptx";
        saxpy_cuda_path      = build_dir + "/cuda_saxpy.ptx";
        compute_miniptx_path = build_dir + "/miniptx_compute_heavy.ptx";
        compute_cuda_path    = build_dir + "/cuda_compute_heavy.ptx";
        compute_nofma_path   = build_dir + "/cuda_compute_heavy_nofma.ptx";
    } else if (argc >= 3) {
        saxpy_miniptx_path = argv[1];
        saxpy_cuda_path    = argv[2];
        if (argc >= 5) {
            compute_miniptx_path = argv[3];
            compute_cuda_path    = argv[4];
        }
        if (argc >= 6) {
            compute_nofma_path   = argv[5];
        }
    } else {
        saxpy_miniptx_path   = build_dir + "/miniptx_saxpy.ptx";
        saxpy_cuda_path      = build_dir + "/cuda_saxpy.ptx";
        compute_miniptx_path = build_dir + "/miniptx_compute_heavy.ptx";
        compute_cuda_path    = build_dir + "/cuda_compute_heavy.ptx";
        compute_nofma_path   = build_dir + "/cuda_compute_heavy_nofma.ptx";
    }

    std::cout << "=================================================================\n";
    std::cout << " MiniPTX Micro-Benchmark Suite: MiniPTX vs. Hand-Written CUDA    \n";
    std::cout << " Target: NVIDIA Tesla V100-SXM2 (sm_70) via CUDA Driver API      \n";
    std::cout << "=================================================================\n\n";

    CUDA_CHECK_MAIN(cuInit(0));
    int device_count = 0;
    CUDA_CHECK_MAIN(cuDeviceGetCount(&device_count));
    if (device_count == 0) {
        std::cerr << "error: no CUDA devices found.\n";
        return 1;
    }
    CUdevice device;
    CUDA_CHECK_MAIN(cuDeviceGet(&device, 0));
    char dev_name[256];
    CUDA_CHECK_MAIN(cuDeviceGetName(dev_name, sizeof(dev_name), device));
    std::cout << "Selected GPU Device 0: " << dev_name << "\n";

    CUcontext context;
    CUDA_CHECK_MAIN(cuCtxCreate(&context, 0, device));

    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK_MAIN(cuMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU Memory: " << (free_mem / (1024 * 1024)) << " MiB free / "
              << (total_mem / (1024 * 1024)) << " MiB total\n\n";

    std::string saxpy_miniptx = read_file(saxpy_miniptx_path);
    std::string saxpy_cuda    = read_file(saxpy_cuda_path);
    if (!saxpy_miniptx.empty() && !saxpy_cuda.empty()) {
        if (!run_saxpy_benchmark(saxpy_miniptx, saxpy_cuda)) {
            std::cerr << "SAXPY benchmark encountered an error.\n";
            cuCtxDestroy(context);
            return 1;
        }
    }

    if (!compute_miniptx_path.empty() && !compute_cuda_path.empty()) {
        std::string compute_miniptx = read_file(compute_miniptx_path);
        std::string compute_cuda    = read_file(compute_cuda_path);
        std::string compute_nofma   = compute_nofma_path.empty() ? "" : read_file(compute_nofma_path, false);
        if (!compute_miniptx.empty() && !compute_cuda.empty()) {
            if (!run_compute_benchmark(compute_miniptx, compute_cuda, compute_nofma)) {
                std::cerr << "Compute-heavy benchmark encountered an error.\n";
                cuCtxDestroy(context);
                return 1;
            }
        }
    }

    std::cout << "Environment & Methodology Notes:\n";
    std::cout << "  - Measurements taken on a shared DGX system (" << dev_name << ").\n";
    std::cout << "  - Alternating A/B/C testing eliminates dynamic thermal/clock scaling bias.\n";
    std::cout << "  - Pure CUDA Driver API (cuEventElapsedTime) timing strictly isolates GPU kernel execution.\n";

    CUDA_CHECK_MAIN(cuCtxDestroy(context));
    return 0;
}
