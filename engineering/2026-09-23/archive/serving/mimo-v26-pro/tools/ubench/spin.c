#include <stdlib.h>
#include <immintrin.h>
#include <unistd.h>
#include <time.h>
int main(int c, char ** v) { time_t end = time(0) + (c > 1 ? atoi(v[1]) : 600); while (time(0) < end) for (int i = 0; i < 100000; i++) _mm_pause(); return 0; }
