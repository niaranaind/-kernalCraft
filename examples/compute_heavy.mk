kernel compute_heavy(float* x, float* y, float a, int n, int iters) {
    int i = tid.x + bid.x * bdim.x;
    if (i < n) {
        float v = x[i];
        for (int j = 0; j < iters; j = j + 1) {
            v = v * a + 0.5;
        }
        y[i] = v;
    }
}
