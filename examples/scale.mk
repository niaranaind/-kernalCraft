kernel scale(float* x, float s, int n) {
    int i = tid.x + bid.x * bdim.x;
    if (i < n) { x[i] = x[i] * s; }
}
