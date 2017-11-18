# rendezvous-dz

Instruções de Compilação

1- Clonar o repositório <br />
2- Inserir o arquivo de entrada com o nome "in.dat" nas pastas rendezvous-dz/serial e rendezvous-dz/openmp.

Compilando o Código Serial:
```
make serial

cd serial

main <numeroDePosicoesIniciais>

```
Compilando o Código Serial para Profiling:
```
make serial-gprof

cd serial

main <numeroDePosicoesIniciais>

```
Compilando o Código paralelo:
```
make openmp

cd openmp

main <numeroDePosicoesIniciais> <numeroDeThreads>

```
