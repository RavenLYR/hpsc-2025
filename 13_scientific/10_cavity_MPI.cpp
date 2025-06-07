#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <fstream>
using namespace std;

typedef vector<vector<float>> matrix;

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int nx = 41, ny = 41, nt = 500, nit = 50;
  const double dx = 2.0 / (nx - 1), dy = 2.0 / (ny - 1);
  const double dt = 0.01, rho = 1.0, nu = 0.02;

  const int ny_local = ny / size;
  const int ny_total = ny_local + 2;

  auto zeros = [nx](int ny) { return matrix(ny, vector<float>(nx, 0.0f)); };
  matrix u = zeros(ny_total), v = zeros(ny_total), p = zeros(ny_total), b = zeros(ny_total);
  matrix un = zeros(ny_total), vn = zeros(ny_total), pn = zeros(ny_total);

  ofstream ufile, vfile, pfile;
  if (rank == 0) {
    ufile.open("u.dat");
    vfile.open("v.dat");
    pfile.open("p.dat");
  }

  for (int n = 0; n < nt; n++) {
    // 1. Save current velocity field
    un = u;
    vn = v;

    // 2. Halo exchange for un and vn before computing b
    if (rank > 0) {
      MPI_Sendrecv(&un[1][0], nx, MPI_FLOAT, rank - 1, 20,
                   &un[0][0], nx, MPI_FLOAT, rank - 1, 21,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Sendrecv(&vn[1][0], nx, MPI_FLOAT, rank - 1, 22,
                   &vn[0][0], nx, MPI_FLOAT, rank - 1, 23,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank < size - 1) {
      MPI_Sendrecv(&un[ny_local][0], nx, MPI_FLOAT, rank + 1, 21,
                   &un[ny_local+1][0], nx, MPI_FLOAT, rank + 1, 20,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Sendrecv(&vn[ny_local][0], nx, MPI_FLOAT, rank + 1, 23,
                   &vn[ny_local+1][0], nx, MPI_FLOAT, rank + 1, 22,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // 3. Compute b using un and vn
    for (int j = 1; j <= ny_local; j++) {
      for (int i = 1; i < nx - 1; i++) {
        b[j][i] = rho * (
          1. / dt * ((un[j][i+1] - un[j][i-1]) / (2*dx) + (vn[j+1][i] - vn[j-1][i]) / (2*dy)) -
          pow((un[j][i+1] - un[j][i-1]) / (2*dx), 2) -
          2 * ((un[j+1][i] - un[j-1][i]) / (2*dy) * (vn[j][i+1] - vn[j][i-1]) / (2*dx)) -
          pow((vn[j+1][i] - vn[j-1][i]) / (2*dy), 2)
        );
      }
    }

    // 4. Pressure Poisson iteration
    for (int it = 0; it < nit; it++) {
      pn = p;
      for (int j = 1; j <= ny_local; j++) {
        for (int i = 1; i < nx - 1; i++) {
          p[j][i] = (
            (pn[j][i+1] + pn[j][i-1]) * dy * dy +
            (pn[j+1][i] + pn[j-1][i]) * dx * dx -
            b[j][i] * dx * dx * dy * dy
          ) / (2 * (dx * dx + dy * dy));
        }
      }

      if (rank > 0)
        MPI_Sendrecv(&p[1][0], nx, MPI_FLOAT, rank - 1, 0,
                     &p[0][0], nx, MPI_FLOAT, rank - 1, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (rank < size - 1)
        MPI_Sendrecv(&p[ny_local][0], nx, MPI_FLOAT, rank + 1, 1,
                     &p[ny_local+1][0], nx, MPI_FLOAT, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // 5. Compute new u and v
    for (int j = 1; j <= ny_local; j++) {
      for (int i = 1; i < nx - 1; i++) {
        u[j][i] = un[j][i] -
          un[j][i] * dt / dx * (un[j][i] - un[j][i-1]) -
          vn[j][i] * dt / dy * (un[j][i] - un[j-1][i]) -
          dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1]) +
          nu * dt / (dx*dx) * (un[j][i+1] - 2*un[j][i] + un[j][i-1]) +
          nu * dt / (dy*dy) * (un[j+1][i] - 2*un[j][i] + un[j-1][i]);

        v[j][i] = vn[j][i] -
          un[j][i] * dt / dx * (vn[j][i] - vn[j][i-1]) -
          vn[j][i] * dt / dy * (vn[j][i] - vn[j-1][i]) -
          dt / (2 * rho * dy) * (p[j+1][i] - p[j-1][i]) +
          nu * dt / (dx*dx) * (vn[j][i+1] - 2*vn[j][i] + vn[j][i-1]) +
          nu * dt / (dy*dy) * (vn[j+1][i] - 2*vn[j][i] + vn[j-1][i]);
      }
    }

    // 6. Apply lid boundary condition on physical top boundary
    if (rank == size - 1) {
      for (int i = 0; i < nx; i++) {
        u[ny_local][i] = 1.0;
        v[ny_local][i] = 0.0;
      }
    }

    // 7. Output every 10 steps
    if (n % 10 == 0) {
      vector<float> u_local_flat, v_local_flat, p_local_flat;
      for (int j = 1; j <= ny_local; j++) {
        for (int i = 0; i < nx; i++) {
          u_local_flat.push_back(u[j][i]);
          v_local_flat.push_back(v[j][i]);
          p_local_flat.push_back(p[j][i]);
        }
      }

      vector<float> u_global, v_global, p_global;
      if (rank == 0) {
        u_global.resize(nx * ny);
        v_global.resize(nx * ny);
        p_global.resize(nx * ny);
      }

      MPI_Gather(u_local_flat.data(), nx * ny_local, MPI_FLOAT,
                 u_global.data(), nx * ny_local, MPI_FLOAT,
                 0, MPI_COMM_WORLD);
      MPI_Gather(v_local_flat.data(), nx * ny_local, MPI_FLOAT,
                 v_global.data(), nx * ny_local, MPI_FLOAT,
                 0, MPI_COMM_WORLD);
      MPI_Gather(p_local_flat.data(), nx * ny_local, MPI_FLOAT,
                 p_global.data(), nx * ny_local, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

      if (rank == 0) {
        for (int j = 0; j < ny; j++)
          for (int i = 0; i < nx; i++)
            ufile << u_global[j * nx + i] << " ";
        ufile << "\n";
        for (int j = 0; j < ny; j++)
          for (int i = 0; i < nx; i++)
            vfile << v_global[j * nx + i] << " ";
        vfile << "\n";
        for (int j = 0; j < ny; j++)
          for (int i = 0; i < nx; i++)
            pfile << p_global[j * nx + i] << " ";
        pfile << "\n";
      }
    }
  }

  if (rank == 0) {
    ufile.close();
    vfile.close();
    pfile.close();
  }

  MPI_Finalize();
  return 0;
}
