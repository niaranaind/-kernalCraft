// Hand-written CUDA compute-heavy benchmark kernel
// Compiled with: nvcc -ccbin <ccbin> -ptx -arch=sm_70 -O3 examples/cuda_compute_heavy.cu -o build_dgx/cuda_compute_heavy.ptx

extern "C" __global__ void compute_heavy(const float* __restrict__ x, float* __restrict__ y, float a, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        #pragma unroll 1
        for (int j = 0; j < iters; ++j) {
            v = v * a + 0.5f;
        }
        y[i] = v;
    }
}
