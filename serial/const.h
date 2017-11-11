#ifndef CONST_H
#define CONST_H

#include <math.h>

// Altitude em relação à superfice da terra
extern int alt;

// Velocidade angular
extern double w;

// O quadrado da velocidade angular: w^2
extern double ww;

// Limite máximo para os somatórios
extern int N;

// Tempo máximo a simular
extern int Tmax;

// Passo de tempo de simulação
extern int deltaT;

// Constantes usadas pela equação do dZ
double brute_H (double z0, double gama, double vex);
double brute_I(double zl0, double gama, double X, double vez);

#endif
