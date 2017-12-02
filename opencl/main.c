#include <stdio.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"

#define MAX_SOURCE_SIZE (0x100000)

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
    int          err;               // error code returned from OpenCL calls
    int          count_gama = 17;
    int          count_ve = 10;
    int          count_X = 100;
    int          count_jn = count_gama * count_X * count_ve * 20;
    int          count_I = count_gama * count_X * count_ve;
    int          count_H = count_gama * count_ve;
    float        w = 398600.4418/sqrt((6378.0 + 220)*(6378.0 + 220)*(6378.0 + 220));
    float*       h_gama = (float*) calloc(count_gama, sizeof(float));
    float*       h_ve = (float*) calloc(count_ve, sizeof(float));
    float*       h_X = (float*) calloc(count_X, sizeof(float));
    float*       h_jn = (float*) calloc(count_jn, sizeof(float));
    float*       h_I = (float*) calloc(count_I, sizeof(float));
    float*       h_H = (float*) calloc(count_H, sizeof(float));
    float*       h_w = &w;

    // Carregando o codigo fonte dos kernels
    const char *HKernelSource = src_to_str("brute_H.cl");
    const char *IKernelSource = src_to_str("brute_I.cl");
    const char *JnKernelSource = src_to_str("brute_jn.cl");

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

    cl_device_id     device_id;     // compute device id
    cl_context       context;       // compute context
    cl_command_queue jn_commands;      // compute command queue
    cl_command_queue I_commands;
    cl_command_queue H_commands;
    cl_program       jn_program;       // compute program
    cl_program       I_program;
    cl_program       H_program;
    cl_kernel        ko_jn;       // compute kernel
    cl_kernel        ko_I;       // compute kernel
    cl_kernel        ko_H;       // compute kernel

    cl_mem d_gama;
    cl_mem d_ve;
    cl_mem d_X;
    cl_mem d_w;
    cl_mem d_jn;
    cl_mem d_I;
    cl_mem d_H;

    // Set up platform and GPU device

    cl_uint numPlatforms;

    // Find number of platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "Finding platforms");
    if (numPlatforms == 0)
    {
        printf("Found 0 platforms!\n");
        return EXIT_FAILURE;
    }

    // Get all platforms
    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    checkError(err, "Getting platforms");

    // Secure a GPU
    for (i = 1; i < numPlatforms; i++)
    {
        err = clGetDeviceIDs(Platform[i], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
        if (err == CL_SUCCESS)
        {
            break;
        }
    }

    if (device_id == NULL)
        checkError(err, "Finding a device");

    err = output_device_info(device_id);
    checkError(err, "Printing device output");

    // Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    checkError(err, "Creating context");

    // Create a command queue
    jn_commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Creating command queue");

    I_commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Creating command queue");

    H_commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Creating command queue");

    // Create the compute program from the source buffer
    jn_program = clCreateProgramWithSource(context, 1, (const char **) & JnKernelSource, NULL, &err);
    checkError(err, "Creating program");

    I_program = clCreateProgramWithSource(context, 1, (const char **) & IKernelSource, NULL, &err);
    checkError(err, "Creating program");

    H_program = clCreateProgramWithSource(context, 1, (const char **) & HKernelSource, NULL, &err);
    checkError(err, "Creating program");

    // Build the program
    err = clBuildProgram(jn_program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to jn build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(jn_program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    err = clBuildProgram(I_program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build I program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(I_program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    err = clBuildProgram(H_program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build H program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(H_program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Create the compute kernel from the program
    ko_jn = clCreateKernel(jn_program, "brute_Jn", &err);
    checkError(err, "Creating kernel");

    // Create the compute kernel from the program
    ko_I = clCreateKernel(I_program, "brute_I", &err);
    checkError(err, "Creating kernel");

    // Create the compute kernel from the program
    ko_H = clCreateKernel(H_program, "brute_H", &err);
    checkError(err, "Creating kernel");

    // Create the input (a, b) and output (c) arrays in device memory
    d_X  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count_X, NULL, &err);
    checkError(err, "Creating buffer d_X");

    d_ve  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count_ve, NULL, &err);
    checkError(err, "Creating buffer d_ve");

    d_gama  = clCreateBuffer(context,  CL_MEM_READ_ONLY, sizeof(float) * count_gama, NULL, &err);
    checkError(err, "Creating buffer d_gama");

    d_w  = clCreateBuffer(context,  CL_MEM_READ_ONLY, sizeof(float), NULL, &err);
    checkError(err, "Creating buffer d_gama");

    d_jn  = clCreateBuffer(context,  CL_MEM_READ_ONLY, sizeof(float) * count_jn, NULL, &err);
    checkError(err, "Creating buffer d_jn");

    d_I = clCreateBuffer(context,  CL_MEM_READ_ONLY, sizeof(float) * count_I, NULL, &err);
    checkError(err, "Creating buffer d_I");

    d_H = clCreateBuffer(context,  CL_MEM_READ_ONLY, sizeof(float) * count_H, NULL, &err);
    checkError(err, "Creating buffer d_H");

    // Write a and b vectors into compute device memory
    err = clEnqueueWriteBuffer(jn_commands, d_X, CL_TRUE, 0, sizeof(float) * count_X, h_X, 0, NULL, NULL);
    checkError(err, "Copying h_X to device at d_X");

    err = clEnqueueWriteBuffer(jn_commands, d_ve, CL_TRUE, 0, sizeof(float) * count_ve, h_ve, 0, NULL, NULL);
    checkError(err, "Copying h_ve to device at d_ve");

    err = clEnqueueWriteBuffer(jn_commands, d_gama, CL_TRUE, 0, sizeof(float) * count_gama, h_gama, 0, NULL, NULL);
    checkError(err, "Copying h_gama to device at d_gama");

    err = clEnqueueWriteBuffer(jn_commands, d_w, CL_TRUE, 0, sizeof(float), h_w, 0, NULL, NULL);
    checkError(err, "Copying h_jn to device at d_jn");

    // Set the arguments to our compute kernel
    err  = clSetKernelArg(ko_jn, 0, sizeof(cl_mem), &d_X);
    err |= clSetKernelArg(ko_jn, 1, sizeof(cl_mem), &d_ve);
    err |= clSetKernelArg(ko_jn, 2, sizeof(cl_mem), &d_gama);
    err |= clSetKernelArg(ko_jn, 3, sizeof(cl_mem), &d_w);
    err |= clSetKernelArg(ko_jn, 4, sizeof(cl_mem), &d_jn);

    err |= clSetKernelArg(ko_jn, 5, sizeof(unsigned int), &count_gama);
    err |= clSetKernelArg(ko_jn, 6, sizeof(unsigned int), &count_ve);
    err |= clSetKernelArg(ko_jn, 7, sizeof(unsigned int), &count_X);
    checkError(err, "Setting kernel arguments");

    // Set the arguments to our compute kernel
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
    checkError(err, "Setting kernel arguments");

    // Set the arguments to our compute kernel
    err |= clSetKernelArg(ko_H, 0, sizeof(cl_mem), &d_ve);
    err |= clSetKernelArg(ko_H, 1, sizeof(cl_mem), &d_gama);
    err |= clSetKernelArg(ko_H, 2, sizeof(cl_mem), &d_w);
    err |= clSetKernelArg(ko_H, 3, sizeof(cl_mem), &d_H);
    float z0 = 0.104698;
    err |= clSetKernelArg(ko_H, 4, sizeof(float), &z0);
    err |= clSetKernelArg(ko_H, 5, sizeof(unsigned int), &count_gama);
    err |= clSetKernelArg(ko_H, 6, sizeof(unsigned int), &count_ve);
    checkError(err, "Setting kernel arguments");

    double rtime = wtime();

    // Execute the kernel over the entire range of our 1d input data set
    // letting the OpenCL runtime choose the work-group size
    const size_t global[3] = {count_X, count_ve, count_gama};
    err = clEnqueueNDRangeKernel(jn_commands, ko_jn, 3, NULL, global, NULL, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel");

    // Wait for the commands to complete before stopping the timer
    err = clFinish(jn_commands);
    checkError(err, "Waiting for kernel to finish");

    rtime = wtime() - rtime;
    printf("\nThe kernel ran in %.20lf seconds\n",rtime);

    // Read back the results from the compute device
    err = clEnqueueReadBuffer( jn_commands, d_jn, CL_TRUE, 0, sizeof(float) * count_jn, h_jn, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array!\n%s\n", err_code(err));
        exit(1);
    }

    rtime = wtime();

    // Execute the kernel over the entire range of our 1d input data set
    // letting the OpenCL runtime choose the work-group size
    err = clEnqueueNDRangeKernel(I_commands, ko_I, 3, NULL, global, NULL, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel");

    // Wait for the commands to complete before stopping the timer
    err = clFinish(I_commands);
    checkError(err, "Waiting for kernel to finish");

    rtime = wtime() - rtime;
    printf("\nThe kernel ran in %.20lf seconds\n",rtime);

    // Read back the results from the compute device
    err = clEnqueueReadBuffer( I_commands, d_I, CL_TRUE, 0, sizeof(float) * count_I, h_I, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array!\n%s\n", err_code(err));
        exit(1);
    }

    // Execute the kernel over the entire range of our 1d input data set
    // letting the OpenCL runtime choose the work-group size
    const size_t H_global[2] = {count_ve, count_gama};
    err = clEnqueueNDRangeKernel(H_commands, ko_H, 2, NULL, H_global, NULL, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel");

    // Wait for the commands to complete before stopping the timer
    err = clFinish(H_commands);
    checkError(err, "Waiting for kernel to finish");

    rtime = wtime() - rtime;
    printf("\nThe kernel ran in %.20lf seconds\n",rtime);

    // Read back the results from the compute device
    err = clEnqueueReadBuffer( H_commands, d_H, CL_TRUE, 0, sizeof(float) * count_H, h_H, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array!\n%s\n", err_code(err));
        exit(1);
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

    for (int i = 0; i < count_H; i++) {
        printf("h_H[%d] = %f\n", i, h_H[i]);
        if ((i+1)%20==0) printf("\n");
        getchar();
    }

    return 0;
}
