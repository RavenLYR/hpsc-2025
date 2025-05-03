#include <cstdio>

__global__ void print(void) {
  printf("Hello GPU\n");
}

int main() {
  printf("Hello CPU\n");
  print<<<2,2>>>();
  cudaDeviceSynchronize();
}
