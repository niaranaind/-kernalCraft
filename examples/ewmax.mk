kernel ewmax(float* a, float* b, float* c, int n) {
    int i = tid.x + bid.x * bdim.x;
    if (i < n) {
        float va = a[i];
        float vb = b[i];
        if (va > vb) {
            c[i] = va;
        } else {
            c[i] = vb;
        }
    }
}
