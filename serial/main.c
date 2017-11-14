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

double vZ(int t, double X, double gama, double vez, double H, double I);

int main(int argc, char *argv[]) {
    //Start time
	time(&timer1);
	tm_info = localtime(&timer1);
	strftime(buffer1, 25, "%d/%m/%Y %H:%M:%S", tm_info);
	startTime = getRealTime();

    int NPI = atoi(argv[1]); // numero de posicoes iniciais
    double velociade_final[15999]; // Vetor das posicoes finais pra cada linha
    FILE *arq, *out;
    char url[] = "in.dat";
    arq = fopen(url, "r");
    out = fopen("serial-out.txt", "w");
    double var1;

    for(int np = 1; np <= NPI; np++) {
        if(arq == NULL) {
            printf("Erro, nao foi possivel abrir o arquivo\n");
            exit(EXIT_FAILURE);
        } else {
            fscanf(arq,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
                &var1, &var1, &var1, &x, &y, &z, &var1, &xl0, &yl0, &zl0, &var1, &var1, &var1, &var1, &var1,
                &var1, &var1, &var1, &var1);
        }

        //int j = 0;
        for(double Ve = 0.5; Ve<=5; Ve+=0.5) {
            double vex, vey, vez;
            vex = vey = vez =Ve*Ve/3;

            for(int aux = -14; aux<=2; aux++){
                double gama = pow(10, aux);

                // O H não depende do X (Chi)
                double H = brute_H (z, gama, vex);

                for(int Xaux=1; Xaux<=100; Xaux++) {
                    double X = Xaux;
                    double I = brute_I (zl0, gama, X, vez);
                    double dz = 0;

                    for(int t = 0; t <= Tmax; t++) {
                        dz = vZ(t, X, gama, vez, H, I);
                    }
                    //velociade_final[j] = dz;
                    //j++;
                    //printf("H:%lf \nI:%lf \nvZ:%lf\n", H, I, dz);
                }
            }
        }
    }
    time(&timer2);
    tm_info = localtime(&timer2);
    strftime(buffer2, 25, "%d/%m/%Y %H:%M:%S", tm_info);

    finalTime = getRealTime();
	fprintf(out, "Tempo total de execucao (s): %lf", finalTime-startTime);
    printf("Tempo total de execucao (s): %lf", finalTime-startTime);
    fclose(out);
    return 0;
}

double vZ(int t, double X, double gama,  double vez, double H, double I) {
    //otimizacao
    double wt = w*t;

    double resultJn = 0;
    double result1 = (-H)*w*sin(wt)+I*w*cos(wt);
    double result2 = 0;

    //otimizacao
    double gama_wpow = (gama/w)*(gama/w);
    double gamat = gama*t;

    for (int n = 1; n <= N; n++) {
        //brute_J
        resultJn = vez/(n*pow(X,n)*w)/(1+(n*n*gama_wpow));
        if (n%2 == 0) {
            resultJn = -resultJn;
        }
        //brute_J

        result2 += resultJn*((-n)*gama*pow(M_E, -(n*gamat)));
    }

    return result1  - result2;
}
