# Optimize matrix multiplication step-by-step
## Pre-requests
* GCC
* Intel MKL
## How to build and run
```bash
# Build use g++
g++ *.cpp  -m64 -I${MKLROOT}/include -O3 -mavx2 -fopenmp -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
# Run the code
# ./a.out [N]
# N is the size of the matrices. Could be nothing or an integer.
# For 1024 by 1024 matrices:
./a.out
# For 2048 by 2048 matrices:
./a.out 2048

```
