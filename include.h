#ifndef _Hinclude_H_
#define _Hinclude_H_

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

void init_value(double ** A, int M, int N);
void check_error(double ** A, double ** B, int M, int N, const char * msg);
void matmul_2d_mnk(double ** A, double ** B, double ** C, int M, int K, int N);
void matmul_2d_mkn(double ** A, double ** B, double ** C, int M, int K, int N);
void matmul_2d_mkn_blk(double ** A, double ** B, double ** C, int M, int K, int N);
void matmul_2d_mkn_use_kernel_4(double ** A, double ** B, double ** C, int M, int K, int N);
void matmul_2d_mkn_use_kernel_4_avx2(double ** A, double ** B, double ** C, int M, int K, int N);
void matmul_2d_mkn_use_kernel_4_avx2_opt(double ** A, double ** B, double ** C, int M, int K, int N);

#endif