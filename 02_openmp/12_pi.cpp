#include <cstdio>
#include <omp.h>

int main() {
  double start = omp_get_wtime();
  int n = 1e8;
  double dx = 1. / n;
  double pi = 0;
#pragma omp parallel for reduction(+:pi)
  for (int i=0; i<n; i++) {
    double x = (i + 0.5) * dx;
    pi += 4.0 / (1.0 + x * x) * dx;
  }
  double end = omp_get_wtime();
  printf("%17.15f\n",pi);
  printf("Time: %f seconds\n", end - start);
}
/*
==============================
Performance Comparison Summary
==============================

n = 10
---------
Serial:   0.000004 sec
Parallel: 0.000422 sec

n = 1e6
---------
Serial:   0.007143 sec
Parallel: 0.002475 sec

n = 1e8
---------
Serial:   0.716235 sec
Parallel: 0.201076 sec

Conclusion:
-----------
Parallel version becomes significantly faster when n is large enough.
This is because the overhead of thread management is amortized by the workload.
*/
