#include "include.h"
#include <sys/time.h>
#include "mkl.h"

#define TIMEING_TEST(func, msg, start_tt, end_tt, tt_cost) gettimeofday(&start_tt, NULL); \
                                                           func; \
                                                           gettimeofday(&end_tt, NULL); \
                                                           tt_cost = (end_tt.tv_sec - start_tt.tv_sec) * 1000. + (end_tt.tv_usec - start_tt.tv_usec) / 1000. ;\
                                                           std::cout << " " << msg << " " <<  tt_cost << " ms" << std::endl;

int main(int argc, char * argv[])
{
    struct timeval start_t, end_t;
    double t_cost;

    int M = 1024;
    int K = 1024;
    int N = 1024;
    if (argc == 2)
    {
        M = std::atoi(argv[1]);
    }
    K = M;
    N = M;
    std::printf("A is a (%6d by %6d) matrix.\n", M, K);
    std::printf("B is a (%6d by %6d) matrix.\n", K, N);
    int size_A = M * K;
    int size_B = K * N;
    int size_C = M * N;
    double alpha = 1.;
    double beta = 0.;
    double ** A = new double *[M]();
    double ** B = new double *[K]();
    double * A_mkl = new double[size_A];
    double * B_mkl = new double[size_B];

    double ** C_mnk = new double *[M]();
    double ** C_mkn = new double *[M]();
    double ** C_mkn_blk = new double *[M]();
    double ** C_kernel_4 = new double *[M]();
    double ** C_kernel_4_avx2 = new double *[M]();
    double ** C_kernel_4_avx2_opt = new double *[M]();
    double * C_mkl = new double[size_C];

    for (int i = 0; i <= M - 1; i++)
    {
        A[i] = new double [K]();
    }
    for (int i = 0; i <= K - 1; i++)
    {
        B[i] = new double[N]();
    }
    for (int i = 0; i <= M - 1; i++)
    {
        C_mnk[i] = new double[N]();
        C_mkn[i] = new double[N]();
        C_mkn_blk[i] = new double[N]();
        C_kernel_4[i] = new double[N]();
        C_kernel_4_avx2[i] = new double[N]();
        C_kernel_4_avx2_opt[i] = new double[N]();
    }

    auto init_value_mkl = [](double ** A, double * A_mkl, int M, int N)
    {
        #pragma omp parallel for
        for (int i = 0; i <= M - 1; i++)
        {
            for (int j = 0; j <= N - 1; j++)
            {
                int idx_A_mkl = i * N + j;
                A_mkl[idx_A_mkl] = A[i][j];
            }
        }
        return;
    };

    auto check_error_mkl = [](double ** A, double * B, int M, int N, const char * msg)
    {
        double sum = 0;
        for (int i = 0; i <= M - 1; i++)
        {
            for (int j = 0; j <= N - 1; j++)
            {
                int idx_B = i * N + j;
                sum += (A[i][j] - B[idx_B]) * (A[i][j] - B[idx_B]);
            }
        }
        std::cout << msg << " error: " << sum << std::endl;
        return;
    };

    TIMEING_TEST(init_value(A, M, K), "init A", start_t, end_t, t_cost);
    TIMEING_TEST(init_value(B, K, N), "init B", start_t, end_t, t_cost);
    TIMEING_TEST(init_value_mkl(A, A_mkl, M, K), "init A_mkl", start_t, end_t, t_cost);
    TIMEING_TEST(init_value_mkl(B, B_mkl, K, N), "init A_mkl", start_t, end_t, t_cost);

    TIMEING_TEST(matmul_2d_mnk(A, B, C_mnk, M, K, N), "mnk", start_t, end_t, t_cost);
    TIMEING_TEST(matmul_2d_mkn(A, B, C_mkn, M, K, N), "mkn", start_t, end_t, t_cost);
    TIMEING_TEST(matmul_2d_mkn_blk(A, B, C_mkn_blk, M, K, N), "mkn_blk", start_t, end_t, t_cost);
    TIMEING_TEST(matmul_2d_mkn_use_kernel_4(A, B, C_kernel_4, M, K, N), "mkn_kernel_4", start_t, end_t, t_cost);
    TIMEING_TEST(matmul_2d_mkn_use_kernel_4_avx2(A, B, C_kernel_4_avx2, M, K, N), "mkn_kernel_4_avx2", start_t, end_t, t_cost);
    TIMEING_TEST(matmul_2d_mkn_use_kernel_4_avx2_opt(A, B, C_kernel_4_avx2_opt, M, K, N), "mkn_kernel_4_avx2_opt", start_t, end_t, t_cost);
    TIMEING_TEST(cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A_mkl, K, B_mkl, N, 0, C_mkl, N), "mkl", start_t, end_t, t_cost);

    TIMEING_TEST(check_error(C_mnk, C_mkn, M, N, "mnk and mkn"), "check error", start_t, end_t, t_cost);
    TIMEING_TEST(check_error(C_mnk, C_mkn_blk, M, N, "mnk and mkn_blk"), "check error", start_t, end_t, t_cost);
    TIMEING_TEST(check_error(C_mnk, C_kernel_4, M, N, "mnk and kernel_4"), "check error", start_t, end_t, t_cost);
    TIMEING_TEST(check_error(C_mnk, C_kernel_4_avx2, M, N, "mnk and kernel_4_avx2"), "check error", start_t, end_t, t_cost);
    TIMEING_TEST(check_error(C_mnk, C_kernel_4_avx2_opt, M, N, "mnk and kernel_4_avx2_opt"), "check error", start_t, end_t, t_cost);
    TIMEING_TEST(check_error_mkl(C_mnk, C_mkl, M, N, "mnk and mkl"), "check error", start_t, end_t, t_cost);


    for (int i = 0; i <= M - 1; i++)
    {
        delete[] A[i];
    }
    delete[] A;
    delete[] A_mkl;
    for (int i = 0; i <= K - 1; i++)
    {
        delete[] B[i];
    }
    delete[] B;
    delete[] B_mkl;

    for (int i = 0; i <= M - 1; i++)
    {
        delete[] C_mnk[i];
        delete[] C_mkn[i];
        delete[] C_mkn_blk[i];
        delete[] C_kernel_4[i];
        delete[] C_kernel_4_avx2[i];
        delete[] C_kernel_4_avx2_opt[i];
    }
    delete[] C_mnk;
    delete[] C_mkn;
    delete[] C_mkn_blk;
    delete[] C_kernel_4;
    delete[] C_kernel_4_avx2;
    delete[] C_kernel_4_avx2_opt;
    delete[] C_mkl;
    return 0;
}