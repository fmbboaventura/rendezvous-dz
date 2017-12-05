#include <stdio.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"

extern double wtime();       // returns time since some fixed past point (wtime.c)
extern int output_device_info(cl_device_id );

char* src_to_str(char* src) {
    FILE *fp;
    long lSize;
    char *str;

    fp = fopen ( src , "rb" );
    if( !fp ) printf("Erro ao abrir o arquivo %s\n", src),exit(EXIT_FAILURE);

    fseek( fp , 0L , SEEK_END);
    lSize = ftell( fp );
    rewind( fp );

    str = calloc( 1, lSize+1 );
    if( !str ) fclose(fp),printf("src_to_str: Erro para alocar a memoria\n"),exit(EXIT_FAILURE);

    if( 1!=fread( str , lSize, 1 , fp) )
        fclose(fp),free(str),printf("Erro ao abrir o arquivo %s\n", src),exit(1);

    fclose(fp);
    return str;
}

int main() {
    int       err;               // error code returned from OpenCL calls
    int       count_gama = 17;
    int       count_ve = 10;
    int       count_X = 100;
    int       count_jn = count_gama * count_X * count_ve * 20;
    int       count_I = count_gama * count_X * count_ve;
    int       count_H = count_gama * count_ve;
    int       count_vz = 9 * count_gama * count_ve * 86400; // Serão calculados 132192000 valores de vz por chamada ao kernel
    float     w = 398600.4418/sqrt((6378.0 + 220)*(6378.0 + 220)*(6378.0 + 220));
    float*    h_gama = (float*) calloc(count_gama, sizeof(float));
    float*    h_ve = (float*) calloc(count_ve, sizeof(float));
    float*    h_X = (float*) calloc(count_X, sizeof(float));
    float*    h_jn = (float*) calloc(count_jn, sizeof(float));
    float*    h_I = (float*) calloc(count_I, sizeof(float));
    float*    h_H = (float*) calloc(count_H, sizeof(float));
    float*    h_vz = (float*) calloc(count_vz, sizeof(float));
    float*    h_w = &w;

    // Carregando o codigo fonte dos kernels
    const char *HKernelSource = src_to_str("brute_H.cl");
    const char *IKernelSource = src_to_str("brute_I.cl");
    const char *JnKernelSource = src_to_str("brute_jn.cl");
    const char *VzKernelSource = src_to_str("vz.cl");

    // Inicializa os vetores dos parametros tecnológicos
    int i = 0;
    for(float gama = pow(10,-14); gama<=100; gama = gama*10){
        h_gama[i] = gama;
        printf("gama%d: %f\n", i, h_gama[i]);
        i++;
    }

    i = 0;
    for(float Ve = 0.5; Ve<=5; Ve+=0.5) {
        h_ve[i] = Ve*Ve/3;
        printf("ve%d: %f\n", i, h_ve[i]);
        i++;
    }

    i = 0;
    for (float j = 1; j <= 100; j++) {
        h_X[i] = j;
        printf("X%d: %f\n", i, h_X[i]);
        i++;
    }

    cl_device_id     device_id;
    cl_context       context;
    cl_command_queue jn_commands;
    cl_command_queue I_commands;
    cl_command_queue H_commands;
    cl_command_queue Vz_commands;
    cl_program       jn_program;
    cl_program       I_program;
    cl_program       H_program;
    cl_program       Vz_program;
    cl_kernel        ko_jn;
    cl_kernel        ko_I;
    cl_kernel        ko_H;
    cl_kernel        ko_Vz;

    cl_mem d_gama;
    cl_mem d_ve;
    cl_mem d_X;
    cl_mem d_w;
    cl_mem d_jn;
    cl_mem d_I;
    cl_mem d_H;
    cl_mem d_Vz;

    cl_uint numPlatforms;

    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "Encontrando plataformas");
    if (numPlatforms == 0)
    {
        printf("Nenhuma plataforma encontrada!\n");
        return EXIT_FAILURE;
    }

    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    checkError(err, "Recuperando platforms");

    // Buscando uma GPU
    for (i = 1; i < numPlatforms; i++)
    {
        err = clGetDeviceIDs(Platform[i], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
        if (err == CL_SUCCESS)
        {
            break;
        }
    }

    if (device_id == NULL)
        checkError(err, "Encontrando device");

    err = output_device_info(device_id);
    checkError(err, "printando device output");

    // Criando contexto
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    checkError(err, "Criando contexto");

    // Criando as command queues
    jn_commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Criando command queue para o calculo do Jn");

    I_commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Criando command queue para o calculo do I");

    H_commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Criando command queue para o calculo do H");

    Vz_commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Criando command queue para o calculo do Vz");

    // Criando os programas a partir do codigo fonte
    jn_program = clCreateProgramWithSource(context, 1, (const char **) & JnKernelSource, NULL, &err);
    checkError(err, "Criando programa do kernel brute_jn");

    I_program = clCreateProgramWithSource(context, 1, (const char **) & IKernelSource, NULL, &err);
    checkError(err, "Criando programa do kernel brute_I");

    H_program = clCreateProgramWithSource(context, 1, (const char **) & HKernelSource, NULL, &err);
    checkError(err, "Criando programa do kernel brute_H");

    Vz_program = clCreateProgramWithSource(context, 1, (const char **) & VzKernelSource, NULL, &err);
    checkError(err, "Criando programa do kernel vz");

    // Compilando os programas
    err = clBuildProgram(jn_program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Erro: Falha ao compilar o prgrama do brute_jn!\n%s\n", err_code(err));
        clGetProgramBuildInfo(jn_program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    err = clBuildProgram(I_program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Erro: Falha ao compilar o prgrama do brute_I!\n%s\n", err_code(err));
        clGetProgramBuildInfo(I_program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    err = clBuildProgram(H_program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Erro: Falha ao compilar o prgrama do brute_H\n%s\n", err_code(err));
        clGetProgramBuildInfo(H_program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    err = clBuildProgram(Vz_program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Erro: Falha ao compilar o prgrama do vz!\n%s\n", err_code(err));
        clGetProgramBuildInfo(Vz_program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Usa os programas para criar os kernels
    ko_jn = clCreateKernel(jn_program, "brute_Jn", &err);
    checkError(err, "Criando kernel do brute_jn");

    ko_I = clCreateKernel(I_program, "brute_I", &err);
    checkError(err, "Criando kernel do brute_I");

    ko_H = clCreateKernel(H_program, "brute_H", &err);
    checkError(err, "Criando kernel do brute_H");

    ko_Vz = clCreateKernel(Vz_program, "vz", &err);
    checkError(err, "Criando kernel do vz");

    // Criando os buffers de entrada e saida na memória do device
    d_X  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count_X, NULL, &err);
    checkError(err, "Criando buffer d_X");

    d_ve  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count_ve, NULL, &err);
    checkError(err, "Criando buffer d_ve");

    d_gama  = clCreateBuffer(context,  CL_MEM_READ_ONLY, sizeof(float) * count_gama, NULL, &err);
    checkError(err, "Criando buffer d_gama");

    d_w  = clCreateBuffer(context,  CL_MEM_READ_ONLY, sizeof(float), NULL, &err);
    checkError(err, "Criando buffer d_w");

    d_jn  = clCreateBuffer(context,  CL_MEM_READ_WRITE, sizeof(float) * count_jn, NULL, &err);
    checkError(err, "Criando buffer d_jn");

    d_I = clCreateBuffer(context,  CL_MEM_READ_WRITE, sizeof(float) * count_I, NULL, &err);
    checkError(err, "Criando buffer d_I");

    d_H = clCreateBuffer(context,  CL_MEM_READ_WRITE, sizeof(float) * count_H, NULL, &err);
    checkError(err, "Criando buffer d_H");

    d_Vz = clCreateBuffer(context,  CL_MEM_READ_WRITE, sizeof(float) * count_vz, NULL, &err);
    checkError(err, "Criando buffer d_Vz");

    // Escreve o vetor dos parametros tecnológicos e outras constantes na memoria do device
    err = clEnqueueWriteBuffer(jn_commands, d_X, CL_TRUE, 0, sizeof(float) * count_X, h_X, 0, NULL, NULL);
    checkError(err, "Copiando h_X para o device em d_X");

    err = clEnqueueWriteBuffer(jn_commands, d_ve, CL_TRUE, 0, sizeof(float) * count_ve, h_ve, 0, NULL, NULL);
    checkError(err, "Copiando h_ve para o device em d_ve");

    err = clEnqueueWriteBuffer(jn_commands, d_gama, CL_TRUE, 0, sizeof(float) * count_gama, h_gama, 0, NULL, NULL);
    checkError(err, "Copiando h_gama para o device em d_gama");

    err = clEnqueueWriteBuffer(jn_commands, d_w, CL_TRUE, 0, sizeof(float), h_w, 0, NULL, NULL);
    checkError(err, "Copiando h_jn para o device em d_jn");

    // Definindo os argumentos para o kernel jn
    err  = clSetKernelArg(ko_jn, 0, sizeof(cl_mem), &d_X);
    err |= clSetKernelArg(ko_jn, 1, sizeof(cl_mem), &d_ve);
    err |= clSetKernelArg(ko_jn, 2, sizeof(cl_mem), &d_gama);
    err |= clSetKernelArg(ko_jn, 3, sizeof(cl_mem), &d_w);
    err |= clSetKernelArg(ko_jn, 4, sizeof(cl_mem), &d_jn);

    err |= clSetKernelArg(ko_jn, 5, sizeof(unsigned int), &count_gama);
    err |= clSetKernelArg(ko_jn, 6, sizeof(unsigned int), &count_ve);
    err |= clSetKernelArg(ko_jn, 7, sizeof(unsigned int), &count_X);
    checkError(err, "Definindo os argumentos para o kernel jn");

    // Definindo os argumentos para o kernel I
    err  = clSetKernelArg(ko_I, 0, sizeof(cl_mem), &d_X);
    err |= clSetKernelArg(ko_I, 1, sizeof(cl_mem), &d_ve);
    err |= clSetKernelArg(ko_I, 2, sizeof(cl_mem), &d_gama);
    err |= clSetKernelArg(ko_I, 3, sizeof(cl_mem), &d_w);
    err |= clSetKernelArg(ko_I, 4, sizeof(cl_mem), &d_I);
    err |= clSetKernelArg(ko_I, 5, sizeof(cl_mem), &d_jn);
    float vz0 = -0.000171;
    err |= clSetKernelArg(ko_I, 6, sizeof(float), &vz0);
    err |= clSetKernelArg(ko_I, 7, sizeof(unsigned int), &count_gama);
    err |= clSetKernelArg(ko_I, 8, sizeof(unsigned int), &count_ve);
    err |= clSetKernelArg(ko_I, 9, sizeof(unsigned int), &count_X);
    checkError(err, "// Definindo os argumentos para o kernel I");

    // Definindo os argumentos para o kernel H
    err |= clSetKernelArg(ko_H, 0, sizeof(cl_mem), &d_ve);
    err |= clSetKernelArg(ko_H, 1, sizeof(cl_mem), &d_gama);
    err |= clSetKernelArg(ko_H, 2, sizeof(cl_mem), &d_w);
    err |= clSetKernelArg(ko_H, 3, sizeof(cl_mem), &d_H);
    float z0 = 0.104698;
    err |= clSetKernelArg(ko_H, 4, sizeof(float), &z0);
    err |= clSetKernelArg(ko_H, 5, sizeof(unsigned int), &count_gama);
    err |= clSetKernelArg(ko_H, 6, sizeof(unsigned int), &count_ve);
    checkError(err, "Definindo os argumentos para o kernel H");

    // Definindo os argumentos para o kernel vz
    err  = clSetKernelArg(ko_Vz, 0, sizeof(cl_mem), &d_X);
    err |= clSetKernelArg(ko_Vz, 1, sizeof(cl_mem), &d_ve);
    err |= clSetKernelArg(ko_Vz, 2, sizeof(cl_mem), &d_gama);
    err |= clSetKernelArg(ko_Vz, 3, sizeof(cl_mem), &d_w);
    err |= clSetKernelArg(ko_Vz, 4, sizeof(cl_mem), &d_I);
    err |= clSetKernelArg(ko_Vz, 5, sizeof(cl_mem), &d_jn);
    err |= clSetKernelArg(ko_Vz, 6, sizeof(cl_mem), &d_H);
    err |= clSetKernelArg(ko_Vz, 7, sizeof(cl_mem), &d_Vz);

    err |= clSetKernelArg(ko_Vz, 8, sizeof(unsigned int), &count_gama);
    err |= clSetKernelArg(ko_Vz, 9, sizeof(unsigned int), &count_ve);
    err |= clSetKernelArg(ko_Vz, 10, sizeof(unsigned int), &count_X);
    checkError(err, "Definindo os argumentos para o kernel vz");

    double rtime = wtime();

    // Executa o kernel brute_jn
    const size_t global[3] = {count_X, count_ve, count_gama};
    err = clEnqueueNDRangeKernel(jn_commands, ko_jn, 3, NULL, global, NULL, 0, NULL, NULL);
    checkError(err, "Enfileirando kernel brute_jn");

    // Espera os comandos concluirem para parar o timer
    err = clFinish(jn_commands);
    checkError(err, "Esperando o termino do kernel");

    rtime = wtime() - rtime;
    printf("\nO kernel brute_jn executou em %.20lf segundos\n",rtime);

    // Read back the results from the compute device
    // err = clEnqueueReadBuffer( jn_commands, d_jn, CL_TRUE, 0, sizeof(float) * count_jn, h_jn, 0, NULL, NULL );
    // if (err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to read output array!\n%s\n", err_code(err));
    //     exit(1);
    // }

    // Para calcular o tempo de execução do brute_I
    rtime = wtime();

    // Executa o kernel do brute_I
    err = clEnqueueNDRangeKernel(I_commands, ko_I, 3, NULL, global, NULL, 0, NULL, NULL);
    checkError(err, "Enfileirando kernel brute_I");

    err = clFinish(I_commands);
    checkError(err, "Esperando pelo termino do kernel");

    rtime = wtime() - rtime;
    printf("\nO kernel brute_I executou em %.20lf segundos\n",rtime);

    // Read back the results from the compute device
    // err = clEnqueueReadBuffer( I_commands, d_I, CL_TRUE, 0, sizeof(float) * count_I, h_I, 0, NULL, NULL );
    // if (err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to read output array!\n%s\n", err_code(err));
    //     exit(1);
    // }

    // para o calculo do tempo do brute_H
    rtime = wtime();

    // Executa o kernel brute_H
    const size_t H_global[2] = {count_ve, count_gama};
    err = clEnqueueNDRangeKernel(H_commands, ko_H, 2, NULL, H_global, NULL, 0, NULL, NULL);
    checkError(err, "Enfileirando kernel brute_H");

    // Wait for the commands to complete before stopping the timer
    err = clFinish(H_commands);
    checkError(err, "Esperando pelo termino do kernel");

    rtime = wtime() - rtime;
    printf("\nO kernel brute_H executou em %.20lf segundos\n",rtime);

    rtime = wtime();
    const size_t Vz_global[2] = {7776, count_X};
    err = clEnqueueNDRangeKernel(Vz_commands, ko_Vz, 2, NULL, Vz_global, NULL, 0, NULL, NULL);
    checkError(err, "Enfileirando kernel vz");

    // Wait for the commands to complete before stopping the timer
    err = clFinish(Vz_commands);
    checkError(err, "Esperando pelo termino do kernel");

    rtime = wtime() - rtime;
    printf("\nO kernel vz executou em %.20lf segundos\n",rtime);

    // Read back the results from the compute device
    // err = clEnqueueReadBuffer( H_commands, d_H, CL_TRUE, 0, sizeof(float) * count_H, h_H, 0, NULL, NULL );
    // if (err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to read output array!\n%s\n", err_code(err));
    //     exit(1);
    // }

    // for (size_t i = 0; i < count_jn; i++) {
    //     printf("h_jn[%d] = %f\n", i, h_jn[i]);
    //     if ((i+1)%20==0) printf("\n");//getchar();
    // }

    // for (int i = 0; i < count_I; i++) {
    //     printf("h_I[%d] = %f\n", i, h_I[i]);
    //     if ((i+1)%20==0) printf("\n");
    //     getchar();
    // }

    // for (int i = 0; i < count_H; i++) {
    //     printf("h_H[%d] = %f\n", i, h_H[i]);
    //     if ((i+1)%20==0) printf("\n");
    //     getchar();
    // }

    return 0;
}
