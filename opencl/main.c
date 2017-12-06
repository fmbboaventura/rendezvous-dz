#include <stdio.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"

extern double wtime();
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

int main(int argc, char *argv[]) {
    int       err;
    int       count_gama = 17;
    int       count_ve = 10;
    int       count_X = 100;
    int       count_jn = count_gama * count_X * count_ve * 20;
    int       count_I = count_gama * count_X * count_ve;
    int       count_H = count_gama * count_ve;
    int       count_vz = 9 * count_gama * count_ve * 86400; // Serão calculados 132192000 valores de vz por chamada ao kernel
    float     w = 398600.4418/sqrt((6378.0 + 220)*(6378.0 + 220)*(6378.0 + 220));
    float     z0;
    float     vz0;
    float     p_time = 0;
    float     t_time = wtime();
    float*    h_gama = (float*) calloc(count_gama, sizeof(float));
    float*    h_ve = (float*) calloc(count_ve, sizeof(float));
    float*    h_jn = (float*) calloc(count_jn, sizeof(float));
    float*    h_I = (float*) calloc(count_I, sizeof(float));
    float*    h_H = (float*) calloc(count_H, sizeof(float));
    float*    h_vz = (float*) calloc(count_vz, sizeof(float));
    float*    h_w = &w;

    int NPI = atoi(argv[1]); // numero de posicoes iniciais
    FILE *arq;
    char url[] = "in.dat";
    float var1;
    arq = fopen(url, "r");

    if(arq == NULL) {
        printf("Erro, nao foi possivel abrir o arquivo\n");
        exit(EXIT_FAILURE);
    }

    // Carregando o codigo fonte dos kernels
    const char *HKernelSource = src_to_str("brute_H.cl");
    const char *IKernelSource = src_to_str("brute_I.cl");
    const char *JnKernelSource = src_to_str("brute_jn.cl");
    const char *vzKernelSource = src_to_str("vz.cl");

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

    cl_device_id     device_id;
    cl_context       context;
    cl_command_queue jn_commands;
    cl_command_queue I_commands;
    cl_command_queue H_commands;
    cl_command_queue vz_commands;
    cl_program       jn_program;
    cl_program       I_program;
    cl_program       H_program;
    cl_program       vz_program;
    cl_kernel        ko_jn;
    cl_kernel        ko_I;
    cl_kernel        ko_H;
    cl_kernel        ko_vz;

    cl_mem d_gama;
    cl_mem d_ve;
    cl_mem d_w;
    cl_mem d_jn;
    cl_mem d_I;
    cl_mem d_H;
    cl_mem d_vz;

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

    vz_commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Criando command queue para o calculo do vz");

    // Criando os programas a partir do codigo fonte
    jn_program = clCreateProgramWithSource(context, 1, (const char **) & JnKernelSource, NULL, &err);
    checkError(err, "Criando programa do kernel brute_jn");

    I_program = clCreateProgramWithSource(context, 1, (const char **) & IKernelSource, NULL, &err);
    checkError(err, "Criando programa do kernel brute_I");

    H_program = clCreateProgramWithSource(context, 1, (const char **) & HKernelSource, NULL, &err);
    checkError(err, "Criando programa do kernel brute_H");

    vz_program = clCreateProgramWithSource(context, 1, (const char **) & vzKernelSource, NULL, &err);
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

    err = clBuildProgram(vz_program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Erro: Falha ao compilar o prgrama do vz!\n%s\n", err_code(err));
        clGetProgramBuildInfo(vz_program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
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

    ko_vz = clCreateKernel(vz_program, "vz", &err);
    checkError(err, "Criando kernel do vz");

    // Criando os buffers de entrada e saida na memória do device
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

    d_vz = clCreateBuffer(context,  CL_MEM_READ_WRITE, sizeof(float) * count_vz, NULL, &err);
    checkError(err, "Criando buffer d_vz");

    // Escreve o vetor dos parametros tecnológicos e outras constantes na memoria do device
    err = clEnqueueWriteBuffer(jn_commands, d_ve, CL_TRUE, 0, sizeof(float) * count_ve, h_ve, 0, NULL, NULL);
    checkError(err, "Copiando h_ve para o device em d_ve");

    err = clEnqueueWriteBuffer(jn_commands, d_gama, CL_TRUE, 0, sizeof(float) * count_gama, h_gama, 0, NULL, NULL);
    checkError(err, "Copiando h_gama para o device em d_gama");

    err = clEnqueueWriteBuffer(jn_commands, d_w, CL_TRUE, 0, sizeof(float), h_w, 0, NULL, NULL);
    checkError(err, "Copiando h_jn para o device em d_jn");

    // Definindo os argumentos para o kernel jn
    err |= clSetKernelArg(ko_jn, 0, sizeof(cl_mem), &d_ve);
    err |= clSetKernelArg(ko_jn, 1, sizeof(cl_mem), &d_gama);
    err |= clSetKernelArg(ko_jn, 2, sizeof(cl_mem), &d_w);
    err |= clSetKernelArg(ko_jn, 3, sizeof(cl_mem), &d_jn);

    err |= clSetKernelArg(ko_jn, 4, sizeof(unsigned int), &count_gama);
    err |= clSetKernelArg(ko_jn, 5, sizeof(unsigned int), &count_ve);
    err |= clSetKernelArg(ko_jn, 6, sizeof(unsigned int), &count_X);
    checkError(err, "Definindo os argumentos para o kernel jn");

    // Definindo os argumentos para o kernel vz
    err |= clSetKernelArg(ko_vz, 0, sizeof(cl_mem), &d_ve);
    err |= clSetKernelArg(ko_vz, 1, sizeof(cl_mem), &d_gama);
    err |= clSetKernelArg(ko_vz, 2, sizeof(cl_mem), &d_w);
    err |= clSetKernelArg(ko_vz, 3, sizeof(cl_mem), &d_I);
    err |= clSetKernelArg(ko_vz, 4, sizeof(cl_mem), &d_jn);
    err |= clSetKernelArg(ko_vz, 5, sizeof(cl_mem), &d_H);
    err |= clSetKernelArg(ko_vz, 6, sizeof(cl_mem), &d_vz);

    err |= clSetKernelArg(ko_vz, 7, sizeof(unsigned int), &count_gama);
    err |= clSetKernelArg(ko_vz, 8, sizeof(unsigned int), &count_ve);
    err |= clSetKernelArg(ko_vz, 9, sizeof(unsigned int), &count_X);
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
    p_time += rtime;
    printf("\nO kernel brute_jn executou em %.20lf segundos\n",rtime);

    // Read back the results from the compute device
    // err = clEnqueueReadBuffer( jn_commands, d_jn, CL_TRUE, 0, sizeof(float) * count_jn, h_jn, 0, NULL, NULL );
    // if (err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to read output array!\n%s\n", err_code(err));
    //     exit(1);
    // }

    for(int np = 1; np <= NPI; np++){

        // Lê a posição e a velocidade inical do arquivo de entrada;
        fscanf(arq,"%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
            &var1, &var1, &var1, &var1, &var1, &z0, &var1, &var1, &var1, &vz0, &var1, &var1, &var1, &var1, &var1,
            &var1, &var1, &var1, &var1);

        // Definindo os argumentos para o kernel I
        err |= clSetKernelArg(ko_I, 0, sizeof(cl_mem), &d_ve);
        err |= clSetKernelArg(ko_I, 1, sizeof(cl_mem), &d_gama);
        err |= clSetKernelArg(ko_I, 2, sizeof(cl_mem), &d_w);
        err |= clSetKernelArg(ko_I, 3, sizeof(cl_mem), &d_I);
        err |= clSetKernelArg(ko_I, 4, sizeof(cl_mem), &d_jn);
        //float vz0 = -0.000171;
        err |= clSetKernelArg(ko_I, 5, sizeof(float), &vz0);
        err |= clSetKernelArg(ko_I, 6, sizeof(unsigned int), &count_gama);
        err |= clSetKernelArg(ko_I, 7, sizeof(unsigned int), &count_ve);
        err |= clSetKernelArg(ko_I, 8, sizeof(unsigned int), &count_X);
        checkError(err, "// Definindo os argumentos para o kernel I");

        // Definindo os argumentos para o kernel H
        err |= clSetKernelArg(ko_H, 0, sizeof(cl_mem), &d_ve);
        err |= clSetKernelArg(ko_H, 1, sizeof(cl_mem), &d_gama);
        err |= clSetKernelArg(ko_H, 2, sizeof(cl_mem), &d_w);
        err |= clSetKernelArg(ko_H, 3, sizeof(cl_mem), &d_H);
        //float z0 = 0.104698;
        err |= clSetKernelArg(ko_H, 4, sizeof(float), &z0);
        err |= clSetKernelArg(ko_H, 5, sizeof(unsigned int), &count_gama);
        err |= clSetKernelArg(ko_H, 6, sizeof(unsigned int), &count_ve);
        checkError(err, "Definindo os argumentos para o kernel H");

        // Para calcular o tempo de execução do brute_I
        rtime = wtime();

        // Executa o kernel do brute_I
        err = clEnqueueNDRangeKernel(I_commands, ko_I, 3, NULL, global, NULL, 0, NULL, NULL);
        checkError(err, "Enfileirando kernel brute_I");

        err = clFinish(I_commands);
        checkError(err, "Esperando pelo termino do kernel");

        rtime = wtime() - rtime;
        p_time += rtime;
        printf("\nLinha %d: O kernel brute_I executou em %.20lf segundos\n",np, rtime);

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
        p_time += rtime;
        printf("\nLinha %d: O kernel brute_H executou em %.20lf segundos\n",np, rtime);

        size_t vz_global[2] = {7776, count_X};
        size_t vz_offset[2] = {0, 0};
        char c;
        int offset;

        for (offset = 0; offset < 85536; offset += 7776) {
            c = 0;
            vz_offset[0] = offset;
            rtime = wtime();
            err = clEnqueueNDRangeKernel(vz_commands, ko_vz, 2, vz_offset, vz_global, NULL, 0, NULL, NULL);
            checkError(err, "Enfileirando kernel vz");

            // Wait for the commands to complete before stopping the timer
            err = clFinish(vz_commands);
            checkError(err, "Esperando pelo termino do kernel");

            rtime = wtime() - rtime;
            p_time += rtime;
            printf("\nLinha %d - Offset %d: O kernel vz executou em %.20lf segundos\n", np, offset, rtime);

            //Read back the results from the compute device
            err = clEnqueueReadBuffer( vz_commands, d_vz, CL_TRUE, 0, sizeof(float) * count_vz, h_vz, 0, NULL, NULL );
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to read output array!\n%s\n", err_code(err));
                exit(1);
            }

            // printf("Offset: %d. h_vz[%d] = %f\n", offset, 16999, h_vz[16999]);
            // for (int i = 0; i < count_vz; i++) {
            //         if (c != '1'){
            //             printf("Offset: %d. h_vz[%d] = %f\n", offset, i, h_vz[i]);
            //             c = getchar();
            //         }//printf("\n");//getchar();
            // }
        }

        // calcula vz para o resto do tempo
        vz_global[0] = 846;
        vz_offset[0] = offset;
        rtime = wtime();
        err = clEnqueueNDRangeKernel(vz_commands, ko_vz, 2, vz_offset, vz_global, NULL, 0, NULL, NULL);
        checkError(err, "Enfileirando kernel vz");

        // Wait for the commands to complete before stopping the timer
        err = clFinish(vz_commands);
        checkError(err, "Esperando pelo termino do kernel");

        rtime = wtime() - rtime;
        p_time += rtime;
        printf("\nLinha %d - Offset %d: O kernel vz executou em %.20lf segundos\n", np, offset, rtime);

        //Read back the results from the compute device
        err = clEnqueueReadBuffer( vz_commands, d_vz, CL_TRUE, 0, sizeof(float) * count_vz, h_vz, 0, NULL, NULL );
        if (err != CL_SUCCESS)
        {
            printf("Erro: falha ao ler o array de saida!\n%s\n", err_code(err));
            exit(1);
        }

        //printf("Offset: %d. h_vz[%d] = %f\n", offset, (14382000-1), h_vz[(14382000-1)]);
        // for (int i = 0; i < 846; i++) {
        //     if (c != '1'){
        //         printf("Offset: %d. h_vz[%d] = %f\n", offset, i, h_vz[i]);
        //         c = getchar();
        //     }//printf("\n");//getchar();
        // }
    }

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
    //
    clReleaseMemObject(d_w);
    clReleaseMemObject(d_gama);
    clReleaseMemObject(d_ve);
    clReleaseMemObject(d_jn);
    clReleaseMemObject(d_H);
    clReleaseMemObject(d_I);
    clReleaseMemObject(d_vz);
    clReleaseCommandQueue(jn_commands);
    clReleaseCommandQueue(I_commands);
    clReleaseCommandQueue(H_commands);
    clReleaseCommandQueue(vz_commands);
    clReleaseProgram(jn_program);
    clReleaseProgram(I_program);
    clReleaseProgram(H_program);
    clReleaseProgram(vz_program);
    clReleaseKernel(ko_jn);
    clReleaseKernel(ko_I);
    clReleaseKernel(ko_H);
    clReleaseKernel(ko_vz);

    t_time = wtime() - t_time;
    printf("\nTempo total de execucao: %.20f\n", t_time);
    printf("Tempo total de execucao dos kernels %.20f\n", p_time);

    return 0;
}
