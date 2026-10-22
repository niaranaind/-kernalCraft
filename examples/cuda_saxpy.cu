// Hand-written idiomatic CUDA SAXPY kernel
// Compiled with: nvcc -ptx -arch=sm_70 -O3 examples/cuda_saxpy.cu -o build_dgx/cuda_saxpy.ptx

extern "C" __global__ void saxpy(float a, const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
