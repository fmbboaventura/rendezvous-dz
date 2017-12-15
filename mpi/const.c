#include "const.h"

// Altitude em relação à superfice da terra
int alt = 220;

// Velocidade angular
double w = 398600.4418/sqrt((6378.0 + 220)*(6378.0 + 220)*(6378.0 + 220));

// O quadrado da velocidade angular: w^2
double ww =
    (398600.4418/sqrt((6378.0 + 220)*(6378.0 + 220)*(6378.0 + 220)))*
    (398600.4418/sqrt((6378.0 + 220)*(6378.0 + 220)*(6378.0 + 220)));

// Limite máximo para os somatórios
int N = 20;

// Tempo máximo a simular
int Tmax = 86400;

// Passo de tempo de simulação
int deltaT = 1;

double brute_H (double z0, double gama, double vex) {
    double result = 0;
    double sum = 0;
    double aux;

    result = z0;
    //otimizacao
    double vexgama = vex*gama;
    double gama_wpow = (gama/w)*(gama/w);
    //Calculo do somatorio
    for (int n = 1; n <= N; n++) {
        aux = ((vexgama)/(pow(gama,n)*(ww)))/(1+(n*n*gama_wpow));
        if (n%2 == 0) {
            aux = -aux;
        }
        sum += aux;
    }
    result += sum;

    return result;
}

double brute_I(double zl0, double gama, double X, double vez) {
    double result = 0;
    double sum = 0;
    double aux;

    result = zl0/w - (vez/w)*(log((X+1)/X));

    //otimizacao
    double gama_wpow = (gama/w)*(gama/w);

    //Calculo do somatorio
    for (int n = 1; n <= N; n++) {
        aux = ((vez)/(n*n*pow(X,n)*w))/(1+(n*n*gama_wpow));
        if (n%2 == 0) {
            aux = -aux;
        }
        sum += aux;
    }

    result += sum;

    return result;
}
