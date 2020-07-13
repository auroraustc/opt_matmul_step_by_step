#include "include.h"
#include <immintrin.h>
#include <avx2intrin.h>

void init_value(double ** A, int M, int N)
{
    srand(time(NULL));
    #pragma omp parallel for
    for (int i = 0; i <= M - 1; i++)
    {
        for (int j = 0; j <= N - 1; j++)
        {
            A[i][j] = (std::rand() + 0.) / RAND_MAX;
        }
    }
    return;
}

void check_error(double ** A, double ** B, int M, int N, const char * msg)
{
    double sum = 0;
    for (int i = 0; i <= M - 1; i++)
    {
        for (int j = 0; j <= N - 1; j++)
        {
            sum += (A[i][j] - B[i][j]) * (A[i][j] - B[i][j]);
        }
    }
    std::cout << msg << " error: " << sum << std::endl;
    return;
}

void matmul_2d_mnk(double ** A, double ** B, double ** C, int M, int K, int N)
{
    #pragma omp parallel for
    for (int m = 0; m <= M - 1; m++)
    {
        for (int n = 0; n <= N - 1; n++)
        {
            for (int k = 0; k <= K - 1; k++)
            {
                C[m][n] += A[m][k] * B[k][n];
            }
        }
    }
    return;
}

void matmul_2d_mkn(double ** A, double ** B, double ** C, int M, int K, int N)
{
    #pragma omp parallel for
    for (int m = 0; m <= M - 1; m++)
    {
        for (int k = 0; k <= K - 1; k++)
        {
            for (int n = 0; n <= N - 1; n++)
            {
                C[m][n] += A[m][k] * B[k][n];
            }
        }
    }
    return;
}

void do_blk(double ** A, double ** B, double ** C, int M_, int K_, int N_, int blk_size)
{
    for (int m = 0; m <= blk_size - 1; m++)
    {
        for (int k = 0; k <= blk_size - 1; k++)
        {
            for (int n = 0; n <= blk_size - 1; n++)
            {
                C[m + M_][n + N_] += A[m + M_][k + K_] * B[k + K_][n + N_];
            }
        }
    }
    return;
}

void matmul_2d_mkn_blk(double ** A, double ** B, double ** C, int M, int K, int N)
{
    int blk_size = 8;
    #pragma omp parallel for
    for (int m = 0; m <= M - 1; m += blk_size)
    {
        for (int k = 0; k <= K - 1; k += blk_size)
        {
            for (int n = 0; n <= N - 1; n += blk_size)
            {
                do_blk(A, B, C, m, k, n, blk_size);
                // for (int blk_m = m; blk_m <= std::min(m + blk_size - 1, M); blk_m++)
                // {
                //     for (int blk_k = k; blk_k <= std::min(k + blk_size - 1, K); blk_k++)
                //     {
                //         for (int blk_n = n; blk_n <= std::min(n + blk_size - 1, N); blk_n++)
                //         {
                //             C[blk_m][blk_n] += A[blk_m][blk_k] * B[blk_k][blk_n];
                //         }
                //     }
                // }
            }
        }
    }
    return;
}

/* A is a (M x K) matrix, B is a (K x 4) matrix ! */
void matmul_2d_kernel_4(double ** A, double ** B, double ** C, int row_C, int col_C, int K)
{
    register double c0(0), \
                    c1(0), \
                    c2(0), \
                    c3(0), \
                    c4(0), \
                    c5(0), \
                    c6(0), \
                    c7(0), \
                    c8(0), \
                    c9(0), \
                    c10(0), \
                    c11(0), \
                    c12(0), \
                    c13(0), \
                    c14(0), \
                    c15(0);
    double * a0(A[row_C + 0]), * b0(B[0]), \
           * a1(A[row_C + 1]), * b1(B[1]), \
           * a2(A[row_C + 2]), * b2(B[2]), \
           * a3(A[row_C + 3]), * b3(B[3]), \
           * end(a0 + K);
    
    do
    {
        c0 += *(a0) * *(b0);
        c1 += *(a0) * *(b1);
        c2 += *(a0) * *(b2);
        c3 += *(a0++) * *(b3);

        c4 += *(a1) * *(b0);
        c5 += *(a1) * *(b1);
        c6 += *(a1) * *(b2);
        c7 += *(a1++) * *(b3);

        c8 += *(a2) * *(b0);
        c9 += *(a2) * *(b1);
        c10 += *(a2) * *(b2);
        c11 += *(a2++) * *(b3);

        c12 += *(a3) * *(b0++);
        c13 += *(a3) * *(b1++);
        c14 += *(a3) * *(b2++);
        c15 += *(a3++) * *(b3++);
    } while (a0 != end);

    C[row_C + 0][col_C + 0] = c0;
    C[row_C + 0][col_C + 1] = c1;
    C[row_C + 0][col_C + 2] = c2;
    C[row_C + 0][col_C + 3] = c3;

    C[row_C + 1][col_C + 0] = c4;
    C[row_C + 1][col_C + 1] = c5;
    C[row_C + 1][col_C + 2] = c6;
    C[row_C + 1][col_C + 3] = c7;

    C[row_C + 2][col_C + 0] = c8 ;
    C[row_C + 2][col_C + 1] = c9 ;
    C[row_C + 2][col_C + 2] = c10;
    C[row_C + 2][col_C + 3] = c11;

    C[row_C + 3][col_C + 0] = c12;
    C[row_C + 3][col_C + 1] = c13;
    C[row_C + 3][col_C + 2] = c14;
    C[row_C + 3][col_C + 3] = c15;

    return;
}

void matmul_2d_mkn_use_kernel_4(double ** A, double ** B, double ** C, int M, int K, int N)
{
    #pragma omp parallel for
    for (int n = 0; n <= N - 1; n += 4)
    {
        double * tmp[4];
        for (int i = 0; i <= 3; i++)
        {
            tmp[i] = new double[N]();
        }
        for (int k = 0; k <= K - 1; k++)
        {
            tmp[0][k] = B[k][n + 0];
            tmp[1][k] = B[k][n + 1];
            tmp[2][k] = B[k][n + 2];
            tmp[3][k] = B[k][n + 3];
        }
        for (int m = 0; m <= M - 1; m += 4)
        {
            matmul_2d_kernel_4(A, tmp, C, m, n, K);
        }
        for (int i = 0; i <= 3; i++)
        {
            delete[] tmp[i];
        }
    }
    
    return;
}

/* A is a (M x K) matrix, B is a (K x 4) matrix ! */
void matmul_2d_kernel_4_avx2(double ** A, double ** B, double ** C, int row_C, int col_C, int K)
{
    double * a0(A[row_C + 0]), * b0(B[0]), \
           * a1(A[row_C + 1]), * b1(B[1]), \
           * a2(A[row_C + 2]), * b2(B[2]), \
           * a3(A[row_C + 3]), * b3(B[3]), \
           * end(a0 + K);
    
    __m256d c_0, c_1, c_2, c_3, a_0, a_1, a_2, a_3, b_0;
    c_0 = _mm256_set1_pd(0.);
    c_1 = _mm256_set1_pd(0.);
    c_2 = _mm256_set1_pd(0.);
    c_3 = _mm256_set1_pd(0.);
    
    do
    {

        a_0 = _mm256_set1_pd(*(a0++));
        a_1 = _mm256_set1_pd(*(a1++));
        a_2 = _mm256_set1_pd(*(a2++));
        a_3 = _mm256_set1_pd(*(a3++));
        b_0 = _mm256_setr_pd(*(b0++), *(b1++), *(b2++), *(b3++));

        c_0 = _mm256_add_pd(c_0, _mm256_mul_pd(a_0, b_0));
        c_1 = _mm256_add_pd(c_1, _mm256_mul_pd(a_1, b_0));
        c_2 = _mm256_add_pd(c_2, _mm256_mul_pd(a_2, b_0));
        c_3 = _mm256_add_pd(c_3, _mm256_mul_pd(a_3, b_0));
    } while (a0 != end);

    _mm256_storeu_pd(&(C[row_C + 0][col_C]), c_0);
    _mm256_storeu_pd(&(C[row_C + 1][col_C]), c_1);
    _mm256_storeu_pd(&(C[row_C + 2][col_C]), c_2);
    _mm256_storeu_pd(&(C[row_C + 3][col_C]), c_3);

    return;
}

void matmul_2d_mkn_use_kernel_4_avx2(double ** A, double ** B, double ** C, int M, int K, int N)
{
    #pragma omp parallel for
    for (int n = 0; n <= N - 1; n += 4)
    {
        double * tmp[4];
        for (int i = 0; i <= 3; i++)
        {
            tmp[i] = new double[N]();
        }
        for (int k = 0; k <= K - 1; k++)
        {
            tmp[0][k] = B[k][n + 0];
            tmp[1][k] = B[k][n + 1];
            tmp[2][k] = B[k][n + 2];
            tmp[3][k] = B[k][n + 3];
        }
        for (int m = 0; m <= M - 1; m += 4)
        {
            matmul_2d_kernel_4_avx2(A, tmp, C, m, n, K);
        }
        for (int i = 0; i <= 3; i++)
        {
            delete[] tmp[i];
        }
    }

    return;
}

/* A is a (M x K) matrix, B is a (K x 4) matrix ! */
inline double hsum_double_avx(__m256d v)
{
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
            vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
}
inline double hsum2_double_avx(__m256d v)
{
    return v[0] + v[1] + v[2] + v[3];
}

void matmul_2d_kernel_4_avx2_opt(double ** A, double ** B, double ** C, int row_C, int col_C, int K)
{
    double * a0(A[row_C + 0]), * b0(B[0]), \
           * a1(A[row_C + 1]), * b1(B[1]), \
           * a2(A[row_C + 2]), * b2(B[2]), \
           * a3(A[row_C + 3]), * b3(B[3]), \
           * end(a0 + K);
    
    __m256d a_0, a_1, a_2, a_3, b_0, b_1, b_2, b_3, \
            c_0, c_1, c_2, c_3, \
            c_4, c_5, c_6, c_7, \
            c_8, c_9, c_10, c_11, \
            c_12, c_13, c_14, c_15;

    c_0 = _mm256_set1_pd(0.);
    c_1 = _mm256_set1_pd(0.);
    c_2 = _mm256_set1_pd(0.);
    c_3 = _mm256_set1_pd(0.);
    c_4 = _mm256_set1_pd(0.);
    c_5 = _mm256_set1_pd(0.);
    c_6 = _mm256_set1_pd(0.);
    c_7 = _mm256_set1_pd(0.);
    c_8 = _mm256_set1_pd(0.);
    c_9 = _mm256_set1_pd(0.);
    c_10 = _mm256_set1_pd(0.);
    c_11 = _mm256_set1_pd(0.);
    c_12 = _mm256_set1_pd(0.);
    c_13 = _mm256_set1_pd(0.);
    c_14 = _mm256_set1_pd(0.);
    c_15 = _mm256_set1_pd(0.);
    
    do
    {
        a_0 = _mm256_loadu_pd(a0);
        a_1 = _mm256_loadu_pd(a1);
        a_2 = _mm256_loadu_pd(a2);
        a_3 = _mm256_loadu_pd(a3);
        b_0 = _mm256_loadu_pd(b0);
        b_1 = _mm256_loadu_pd(b1);
        b_2 = _mm256_loadu_pd(b2);
        b_3 = _mm256_loadu_pd(b3);

        c_0 = _mm256_add_pd(c_0, _mm256_mul_pd(a_0, b_0));
        c_1 = _mm256_add_pd(c_1, _mm256_mul_pd(a_0, b_1));
        c_2 = _mm256_add_pd(c_2, _mm256_mul_pd(a_0, b_2));
        c_3 = _mm256_add_pd(c_3, _mm256_mul_pd(a_0, b_3));
        c_4 = _mm256_add_pd(c_4, _mm256_mul_pd(a_1, b_0));
        c_5 = _mm256_add_pd(c_5, _mm256_mul_pd(a_1, b_1));
        c_6 = _mm256_add_pd(c_6, _mm256_mul_pd(a_1, b_2));
        c_7 = _mm256_add_pd(c_7, _mm256_mul_pd(a_1, b_3));
        c_8 = _mm256_add_pd(c_8, _mm256_mul_pd(a_2, b_0));
        c_9 = _mm256_add_pd(c_9, _mm256_mul_pd(a_2, b_1));
        c_10 = _mm256_add_pd(c_10, _mm256_mul_pd(a_2, b_2));
        c_11 = _mm256_add_pd(c_11, _mm256_mul_pd(a_2, b_3));
        c_12 = _mm256_add_pd(c_12, _mm256_mul_pd(a_3, b_0));
        c_13 = _mm256_add_pd(c_13, _mm256_mul_pd(a_3, b_1));
        c_14 = _mm256_add_pd(c_14, _mm256_mul_pd(a_3, b_2));
        c_15 = _mm256_add_pd(c_15, _mm256_mul_pd(a_3, b_3));

        a0 += 4;
        a1 += 4;
        a2 += 4;
        a3 += 4;
        b0 += 4;
        b1 += 4;
        b2 += 4;
        b3 += 4;
    } while (a0 != end);

    C[row_C + 0][col_C + 0] = hsum_double_avx(c_0);
    C[row_C + 0][col_C + 1] = hsum_double_avx(c_1);
    C[row_C + 0][col_C + 2] = hsum_double_avx(c_2);
    C[row_C + 0][col_C + 3] = hsum_double_avx(c_3);

    C[row_C + 1][col_C + 0] = hsum_double_avx(c_4);
    C[row_C + 1][col_C + 1] = hsum_double_avx(c_5);
    C[row_C + 1][col_C + 2] = hsum_double_avx(c_6);
    C[row_C + 1][col_C + 3] = hsum_double_avx(c_7);

    C[row_C + 2][col_C + 0] = hsum_double_avx(c_8);
    C[row_C + 2][col_C + 1] = hsum_double_avx(c_9);
    C[row_C + 2][col_C + 2] = hsum_double_avx(c_10);
    C[row_C + 2][col_C + 3] = hsum_double_avx(c_11);

    C[row_C + 3][col_C + 0] = hsum_double_avx(c_12);
    C[row_C + 3][col_C + 1] = hsum_double_avx(c_13);
    C[row_C + 3][col_C + 2] = hsum_double_avx(c_14);
    C[row_C + 3][col_C + 3] = hsum_double_avx(c_15);

    return;
}

void matmul_2d_mkn_use_kernel_4_avx2_opt(double ** A, double ** B, double ** C, int M, int K, int N)
{
    #pragma omp parallel for
    for (int n = 0; n <= N - 1; n += 4)
    {
        double * tmp[4];
        for (int i = 0; i <= 3; i++)
        {
            tmp[i] = new double[N]();
        }
        for (int k = 0; k <= K - 1; k++)
        {
            tmp[0][k] = B[k][n + 0];
            tmp[1][k] = B[k][n + 1];
            tmp[2][k] = B[k][n + 2];
            tmp[3][k] = B[k][n + 3];
        }
        for (int m = 0; m <= M - 1; m += 4)
        {
            matmul_2d_kernel_4_avx2(A, tmp, C, m, n, K);
        }
        for (int i = 0; i <= 3; i++)
        {
            delete[] tmp[i];
        }
    }

    return;
}