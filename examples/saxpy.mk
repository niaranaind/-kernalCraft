kernel saxpy(float a, float* x, float* y, int n) {
    int i = tid.x + bid.x * bdim.x;
    if (i < n) { y[i] = a * x[i] + y[i]; }
}
