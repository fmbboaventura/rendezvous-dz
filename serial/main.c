#include <stdio.h>
#include <stdlib.h>
#include "const.h"
#include <sys/time.h>

double startTime, finalTime;
time_t timer1, timer2;
char buffer1[25], buffer2[25];
struct tm* tm_info;

double getRealTime(){
    struct timeval tm;
    gettimeofday(&tm, NULL);
    return ((double)tm.tv_sec + (double)tm.tv_usec/1000000.0);
}

double x=0, y=0, z=0, xl0=0, yl0=0, zl0=0;

int main(int argc, char *argv[]) {
    printf("Hello\n");
    //Start time
	time(&timer1);
	tm_info = localtime(&timer1);
	strftime(buffer1, 25, "%d/%m/%Y %H:%M:%S", tm_info);
	startTime = getRealTime();

    int NPI = atoi(argv[1]); // numero de posicoes iniciais
    FILE *arq, *out;
    char url[] = "in.dat";
    arq = fopen(url, "r");
    out = fopen("parallel-out.txt", "w");
    double var1;

    //printf("Numero de posicoes iniciais: %d\n", NPI);

    for(int np = 1; np <= NPI; np++) {
        //printf("Problema %d\n", np);
        if(arq == NULL) {
            printf("Erro, nao foi possivel abrir o arquivo\n");
            exit(EXIT_FAILURE);
        } else {
            fscanf(arq,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", &var1, &var1, &var1, &x, &y, &z, &var1, &xl0, &yl0, &zl0, &var1, &var1, &var1, &var1, &var1, &var1, &var1, &var1, &var1);
            //printf("%lf %lf %lf %lf %lf %lf\n", x, y, z, xl0, yl0, zl0);
        }
        //#pragma omp parallel for
        for(double Ve = 0.5; Ve<=0.5; Ve+=0.5) {
            //printf("Ve %d\n", VeAux);
            double vex, vey, vez;
            vex = vey = vez =Ve*Ve/3;
            //#pragma omp parallel for
            for(int aux = 0; aux<=0; aux++){
                //printf("Gama %d\n", aux);
                double gama = pow(10, aux);
                //int tid = omp_get_thread_num();
                //printf("Hello world from omp thread %d\n", tid);
                //#pragma omp parallel for firstprivate(z, x, y, zl0, xl0, yl0)
                for(int Xaux=1; Xaux<=1; Xaux++) {
                    //printf("X %d\n", Xaux);
                    double X = Xaux;

                    double H = brute_H (z, gama, vex);
                    double I = brute_I (zl0, gama, X, vez);

                    printf("H:%lf \nI:%lf \n", H, I);
                    //nave = nave + 1;
                    //int ID = omp_get_thread_num();
                    //printf("Simulando nave %.1f\n", nave);
                    //#pragma omp parallel for
                    for(int t = 1; t <= Tmax; t++) {

                    }
                }
            }
        }
    }
    time(&timer2);
    tm_info = localtime(&timer2);
    strftime(buffer2, 25, "%d/%m/%Y %H:%M:%S", tm_info);

    finalTime = getRealTime();
	fprintf(out, "Tempo em segundos: %lf", finalTime);
    fclose(out);
    return 0;
}
