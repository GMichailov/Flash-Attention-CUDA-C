nvcc -std=c++20 -arch=sm_86 tests/main.cu -o test -diag-suppress=20054
./test