#include <stdio.h>
#include <stdlib.h>
#include "const.h"
#include <mpi.h>
#include <omp.h>
#include <string.h>

extern double wtime();

double x=0, y=0, z=0, xl0=0, yl0=0, zl0=0;

double vZ(int t, double X, double gama, double vez, double H, double I);

int main(int argc, char* argv[]){
	int  my_rank; /* rank do processo */
	int  p;       /* numero de processos */
	int source;   /* rank do sender */
	int dest;     /* rank do receiver */
	int tag=0;    /* tag para messagens */
	char message[100];        /* mensagem */
	MPI_Status status ;

	int base_X;
	int fim_X;
	double base_gama = pow(10,-14);
	double vez;
	int NPI = atoi(argv[1]); // numero de posicoes iniciais
	FILE *arq;
    char url[] = "in.dat";
    double var1;
	double r_time;
	double t_time = wtime();

	arq = fopen(url, "r");
	if(arq == NULL) {
		printf("Erro, nao foi possivel abrir o arquivo\n");
		exit(EXIT_FAILURE);
	}

	/* Inicializa o MPI */
	MPI_Init(&argc, &argv);

	/* pega o process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	/* pega o numero de processos */
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	switch (my_rank) {
		case 0: base_X = 1;  break;
		case 1: base_X = 25; break;
		case 2: base_X = 50; break;
		case 3: base_X = 75; break;
	}
	fim_X = base_X + 25;

	omp_set_num_threads(4);

	r_time = wtime();

	for(int np = 1; np <= NPI; np++) {

		fscanf(arq,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
			&var1, &var1, &var1, &x, &y, &z, &var1, &xl0, &yl0, &zl0, &var1, &var1, &var1, &var1, &var1,
			&var1, &var1, &var1, &var1);

		for(double Ve = 0.5; Ve<=5; Ve+=0.5) {
			vez = Ve*Ve/3;

			for(double gama = base_gama; gama<=100; gama = gama*10){
				// O H não depende do X (Chi)
                double H = brute_H (z, gama, vez);

				for (int X = base_X; X <= fim_X; X++) {
					double I = brute_I (zl0, gama, X, vez);
                    double dz = 0;

					#pragma omp parallel for
					for(int t = 0; t <= Tmax; t++) {
                        dz = vZ(t, X, gama, vez, H, I);
                    }
				}
			}
		}
	}

	r_time = wtime() - r_time;

	if (my_rank !=0){
		/* cria mensagem */
		sprintf(message, "Rank: %d\nTempo de Execucao (s): %f", my_rank, r_time);
		dest = 0;

		MPI_Send(message, strlen(message)+1, MPI_CHAR,
		   dest, tag, MPI_COMM_WORLD);
	}
	else{
		printf("----------------------------\n");
		printf("Numero de Linhas: %d\n", NPI);
		printf("Rank: %d\nTempo de Execucao (s): %f\n", my_rank, r_time);
		printf("Esperando os outros Hosts...\n\n");
		for (source = 1; source < p; source++) {
			MPI_Recv(message, 100, MPI_CHAR, source, tag,
			      MPI_COMM_WORLD, &status);
			printf("%s\n",message);
		}
		t_time = wtime() - t_time;
		printf("Tempo total (s): %f\n", t_time);
		printf("----------------------------\n");
	}


	/* shut down MPI */
	MPI_Finalize();


	return 0;
}

double vZ(int t, double X, double gama,  double vez, double H, double I) {
    //otimizacao
    double wt = w*t;

    double resultJn = 0;
    // Otimizando com a distribuitiva
    double result1 = w*((-H)*sin(wt)+I*cos(wt));
    double result2 = 0;

    //otimizacao
    double gama_wpow = (gama/w)*(gama/w);
    double gamat = gama*t;

    for (int n = 1; n <= N; n++) {
        //brute_J
        resultJn = 1/(n*pow(X,n)*w)/(1+(n*n*gama_wpow));
        if (n%2 == 0) {
            resultJn = -resultJn;
        }
        //brute_J

        result2 += resultJn*((-n)*exp(-(n*gamat)));
    }

    // tirando as constantes do somatorio
    return result1  - vez*gama*result2;
}
