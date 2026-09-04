// Standalone CUDA test: understand prmt.b32 / __byte_perm actual behavior
#include <cstdio>
#include <cuda_runtime.h>

__global__ void test_kernel(unsigned int* out) {
    // Use distinct byte patterns so we can see what's selected
    unsigned int a = 0x04030201u;  // bytes LE: [01, 02, 03, 04]
    unsigned int b = 0x08070605u;  // bytes LE: [05, 06, 07, 08]

    // Test 1: identity — select bytes [0,1,2,3] from a
    out[0] = __byte_perm(a, b, 0x03020100u);
    // Test 2: all byte 0 of a
    out[1] = __byte_perm(a, b, 0x00000000u);
    // Test 3: all byte 3 of a
    out[2] = __byte_perm(a, b, 0x03030303u);
    // Test 4: select bytes [0,1,2,3] from b (indices 4,5,6,7)
    out[3] = __byte_perm(a, b, 0x07060504u);
    // Test 5: select byte 0 of a for all positions
    out[4] = __byte_perm(a, b, 0x00000000u);
    // Test 6: the original failing case
    out[5] = __byte_perm(0xC0804000u, 0xC0804000u, 0x03020100u);
    // Test 7: sel = 0x01010101 (all byte 1 of a)
    out[6] = __byte_perm(a, b, 0x01010101u);
    // Test 8: sel = 0x02020202 (all byte 2 of a)
    out[7] = __byte_perm(a, b, 0x02020202u);
}

int main() {
    unsigned int* d_out;
    cudaMalloc(&d_out, 8 * sizeof(unsigned int));
    test_kernel<<<1, 1>>>(d_out);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
    unsigned int h_out[8];
    cudaMemcpy(h_out, d_out, 8 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("a=0x04030201 b=0x08070605\n");
    printf("Test 1 sel=0x03020100: 0x%08X (expected 0x04030201)\n", h_out[0]);
    printf("Test 2 sel=0x00000000: 0x%08X (expected 0x01010101)\n", h_out[1]);
    printf("Test 3 sel=0x03030303: 0x%08X (expected 0x04040404)\n", h_out[2]);
    printf("Test 4 sel=0x07060504: 0x%08X (expected 0x08070605)\n", h_out[3]);
    printf("Test 5 sel=0x00000000: 0x%08X (expected 0x01010101)\n", h_out[4]);
    printf("Test 6 sel=0x03020100 (C0804000): 0x%08X (expected 0xC0804000)\n", h_out[5]);
    printf("Test 7 sel=0x01010101: 0x%08X (expected 0x02020202)\n", h_out[6]);
    printf("Test 8 sel=0x02020202: 0x%08X (expected 0x03030303)\n", h_out[7]);
    cudaFree(d_out);
    return 0;
}
