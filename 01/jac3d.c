/* Jacobi-3 program */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define L 693
#define ITMAX 100

int main(int an, char **as)
{
    int i, j, k, it;
    double eps;
    double MAXEPS = 0.5f;
    
    double ***A = (double ***)malloc(L * sizeof(double **));
    double ***B = (double ***)malloc(L * sizeof(double **));
    for (i = 0; i < L; i++) {
        A[i] = (double **)malloc(L * sizeof(double *));
        B[i] = (double **)malloc(L * sizeof(double *));
        for (j = 0; j < L; j++) {
            A[i][j] = (double *)malloc(L * sizeof(double));
            B[i][j] = (double *)malloc(L * sizeof(double));
        }
    }

    #pragma omp parallel for private(i,j,k) collapse(3)
    for (i = 0; i < L; i++)
        for (j = 0; j < L; j++)
            for (k = 0; k < L; k++)
            {
                A[i][j][k] = 0.0;
                if (i == 0 || j == 0 || k == 0 || i == L - 1 || j == L - 1 || k == L - 1)
                    B[i][j][k] = 0.0;
                else
                    B[i][j][k] = 4.0 + i + j + k;
            }

    double startt = omp_get_wtime(); 

    /* iteration loop */
    for (it = 1; it <= ITMAX; it++)
    {
        eps = 0.0;

        #pragma omp parallel for private(i,j,k) reduction(max:eps) collapse(3)
        for (i = 1; i < L - 1; i++)
            for (j = 1; j < L - 1; j++)
                for (k = 1; k < L - 1; k++)
                {
                    double tmp = fabs(B[i][j][k] - A[i][j][k]);
                    eps = Max(tmp, eps);
                    A[i][j][k] = B[i][j][k];
                }

        #pragma omp parallel for private(i,j,k) collapse(3)
        for (i = 1; i < L - 1; i++)
            for (j = 1; j < L - 1; j++)
                for (k = 1; k < L - 1; k++)
                    B[i][j][k] = (A[i - 1][j][k] + A[i][j - 1][k] + A[i][j][k - 1] + 
                                  A[i][j][k + 1] + A[i][j + 1][k] + A[i + 1][j][k]) / 6.0f;
        
        printf(" IT = %4d   EPS = %14.7E\n", it, eps);
        if (eps < MAXEPS)
            break;
    }

    double endt = omp_get_wtime(); 

    printf(" Jacobi3D Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", L, L, L);
    printf(" Iterations      =       %12d\n", it-1);
    printf(" Time in seconds =       %12.6lf\n", endt - startt);
    printf(" Operation type  =     floating point\n");
    printf(" END OF Jacobi3D Benchmark\n");

    for (i = 0; i < L; i++) {
        for (j = 0; j < L; j++) {
            free(A[i][j]);
            free(B[i][j]);
        }
        free(A[i]);
        free(B[i]);
    }
    free(A);
    free(B);

    return 0;
}