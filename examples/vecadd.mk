kernel vecadd(int* a, int* b, int* c, int n) {
    int i = tid.x + bid.x * bdim.x;
    if (i < n) { c[i] = a[i] + b[i]; }
}
