#include <stdio.h>
#include <omp.h>

#include <time.h>
#include <stdlib.h>
#include <windows.h>


void foo(int parent_id) {
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    printf("%d:%d:%d -- Nested Thread: %d Parent: %d -- iniciando!\n",
        tm.tm_hour, tm.tm_min, tm.tm_sec, omp_get_thread_num(), parent_id);
    Sleep(10000 + ((1000 * rand())%4001));
    t = time(NULL);
    tm = *localtime(&t);
    printf("%d:%d:%d -- Nested Thread: %d Parent: %d -- terminando!\n",
        tm.tm_hour, tm.tm_min, tm.tm_sec, omp_get_thread_num(), parent_id);
}

int main() {
    srand(time(NULL));   // should only be called once
    printf("Max threads: %d\n", omp_get_max_threads());
    omp_set_nested(1);
    #pragma omp parallel
    {
        time_t t = time(NULL);
        struct tm tm = *localtime(&t);
        int parent_id = omp_get_thread_num();
        printf("%d:%d:%d Thread %d: iniciando!\n", tm.tm_hour, tm.tm_min, tm.tm_sec, parent_id);

        #pragma omp parallel
        {
            foo(parent_id);
        }
        t = time(NULL);
        tm = *localtime(&t);
        printf("%d:%d:%d Thread %d: terminando!\n", tm.tm_hour, tm.tm_min, tm.tm_sec, omp_get_thread_num());
    }
    return 0;
}
