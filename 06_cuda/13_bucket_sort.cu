#include <cstdio>
#include <cstdlib>
#include <vector>

// int main() {
//   int n = 50;
//   int range = 5;
//   std::vector<int> key(n);
//   for (int i=0; i<n; i++) {
//     key[i] = rand() % range;
//     printf("%d ",key[i]);
//   }
//   printf("\n");

//   std::vector<int> bucket(range); 
//   for (int i=0; i<range; i++) {
//     bucket[i] = 0;
//   }
//   for (int i=0; i<n; i++) {
//     bucket[key[i]]++;
//   }
//   for (int i=0, j=0; i<range; i++) {
//     for (; bucket[i]>0; bucket[i]--) {
//       key[j++] = i;
//     }
//   }

//   for (int i=0; i<n; i++) {
//     printf("%d ",key[i]);
//   }
//   printf("\n");
// }
__global__ void generate_keys(int* key, int n, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    key[i] = (i * 37 + 17) % range;  
}
__global__ void init_bucket(int* bucket, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < range)
    bucket[i] = 0;
}
__global__ void count_bucket(int* key, int* bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    atomicAdd(&bucket[key[i]], 1);  
}

int main(){

int* key;
int* bucket;

int n = 50;       
int range = 5;

cudaMallocManaged(&key, n * sizeof(int));
cudaMallocManaged(&bucket, range * sizeof(int));

// Step 1: init
int threads = 128;
int blocks_range = (range + threads - 1) / threads;

init_bucket<<<blocks_range, threads>>>(bucket, range);

generate_keys<<<(n + threads - 1) / threads, threads>>>(key, n, range);

// Step 2: count key
int blocks_data = (n + threads - 1) / threads;
count_bucket<<<blocks_data, threads>>>(key, bucket, n);

cudaDeviceSynchronize();

// Step 3: order
int j = 0;
for (int i = 0; i < range; i++) {
  while (bucket[i]-- > 0) {
    key[j++] = i;
  }
}

// Step 4: verify
for (int i = 0; i < n; i++) {
  printf("%d ", key[i]);
}
printf("\n");

// Step 5: free memory
cudaFree(key);
cudaFree(bucket);

return 0;

}
