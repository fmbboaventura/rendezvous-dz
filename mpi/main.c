#include <stdio.h>
#include <stdlib.h>
#include "const.h"
#include <mpi.h>

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

	int NPI = atoi(argv[1]); // numero de posicoes iniciais
	FILE *arq;
    char url[] = "in.dat";
    double var1;

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

	for(int np = 1; np <= NPI; np++) {

		fscanf(arq,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
			&var1, &var1, &var1, &x, &y, &z, &var1, &xl0, &yl0, &zl0, &var1, &var1, &var1, &var1, &var1,
			&var1, &var1, &var1, &var1);

		switch (my_rank) {
			case 0: printf("Rank: %d z: %f dz: %f\n", my_rank, z, zl0);break;
			case 0: printf("Rank: %d z: %f dz: %f\n", my_rank, z, zl0);break;
			case 0: printf("Rank: %d z: %f dz: %f\n", my_rank, z, zl0);break;
			case 0: printf("Rank: %d z: %f dz: %f\n", my_rank, z, zl0);break;
		}
	}

	// if (my_rank !=0){
	// 	/* create message */
	// 	sprintf(message, "Hello MPI World from process %d!", my_rank);
	// 	dest = 0;
	// 	/* use strlen+1 so that '\0' get transmitted */
	// 	MPI_Send(message, strlen(message)+1, MPI_CHAR,
	// 	   dest, tag, MPI_COMM_WORLD);
    //     system("cat /proc/cpuinfo");
	// }
	// else{
	// 	printf("Hello MPI World From process 0: Num processes: %d\n",p);
	// 	for (source = 1; source < p; source++) {
	// 		MPI_Recv(message, 100, MPI_CHAR, source, tag,
	// 		      MPI_COMM_WORLD, &status);
	// 		printf("%s\n",message);
    //         system("cat /proc/cpuinfo");
	// 	}
	// }

	/* shut down MPI */
	MPI_Finalize();


	return 0;
}
