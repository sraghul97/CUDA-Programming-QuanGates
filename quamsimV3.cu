#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sys/time.h>
#include <inttypes.h>
#include <math.h>
#include <cuda_runtime.h>

using namespace std;
uint64_t InputMatrixSize = 0;

__global__ void QuantumGate(float *A, float *B, float *C, uint64_t Alength, uint64_t Blength)
{
    uint64_t Aposition = 2 * threadIdx.x;
    uint64_t Bposition = blockIdx.x;
    float MatrixResult[4];
    __shared__ float SharedMemMatrix[64];

    if ((Aposition < Alength) && (Bposition < Blength))
    {
        if (Aposition == 0)
        {
            for (uint64_t Index = 0; Index < 64; Index++)
                SharedMemMatrix[Index] = B[(Bposition * 64) + Index];
        }
        __syncthreads();
        for (uint64_t QuBitOperationCount = 0; QuBitOperationCount < 6; QuBitOperationCount++)
        {
            uint64_t QbitPower = 1 << QuBitOperationCount; 
            uint64_t Remainder = Aposition % QbitPower;
            uint64_t Remainder1 = (Aposition + 1) % QbitPower;

            MatrixResult[0] = 0;
            MatrixResult[1] = 0;
            MatrixResult[2] = 0;
            MatrixResult[3] = 0;
            for (uint64_t Iteration = 0; Iteration < 2; Iteration++)
            {
                MatrixResult[0] += A[(QuBitOperationCount * 4) + Iteration + 0] * SharedMemMatrix[(Iteration * QbitPower) + ((Aposition - Remainder) * 2) + Remainder];
                MatrixResult[1] += A[(QuBitOperationCount * 4) + Iteration + 2] * SharedMemMatrix[(Iteration * QbitPower) + ((Aposition - Remainder) * 2) + Remainder];
                MatrixResult[2] += A[(QuBitOperationCount * 4) + Iteration + 0] * SharedMemMatrix[(Iteration * QbitPower) + ((Aposition + 1 - Remainder1) * 2) + Remainder1];
                MatrixResult[3] += A[(QuBitOperationCount * 4) + Iteration + 2] * SharedMemMatrix[(Iteration * QbitPower) + ((Aposition + 1 - Remainder1) * 2) + Remainder1];
            }
            SharedMemMatrix[(0 * QbitPower) + ((Aposition - Remainder) * 2) + Remainder] = MatrixResult[0];
            SharedMemMatrix[(1 * QbitPower) + ((Aposition - Remainder) * 2) + Remainder] = MatrixResult[1];
            SharedMemMatrix[(0 * QbitPower) + ((Aposition + 1 - Remainder1) * 2) + Remainder1] = MatrixResult[2];
            SharedMemMatrix[(1 * QbitPower) + ((Aposition + 1 - Remainder1) * 2) + Remainder1] = MatrixResult[3];
           
            C[(Bposition * 64) + (0 * QbitPower) + ((Aposition - Remainder) * 2) + Remainder] = MatrixResult[0];
            C[(Bposition * 64) + (1 * QbitPower) + ((Aposition - Remainder) * 2) + Remainder] = MatrixResult[1];
            C[(Bposition * 64) + (0 * QbitPower) + ((Aposition + 1 - Remainder1) * 2) + Remainder1] = MatrixResult[2];
            C[(Bposition * 64) + (1 * QbitPower) + ((Aposition + 1 - Remainder1) * 2) + Remainder1] = MatrixResult[3];
            
            __syncthreads();
        } 
    }
    __syncthreads();
}

uint64_t QbitPosition[6];

uint64_t QuBitIndexCalculator(uint64_t IndexCounter)
{
    bool BinaryIndexCounter[6];
    uint64_t Result = 0;
    for (uint64_t counter = 0; counter < 6; counter++)
    {
        uint64_t QbitVariable = pow(2, QbitPosition[counter]);
        BinaryIndexCounter[counter] = IndexCounter & (1 << counter);
        if (BinaryIndexCounter[counter])
            Result += pow(2, QbitPosition[counter]);
    }
    // cout << "QuBitIndexCalculator" << Result << '\t' << IndexCounter << '\t' << BinaryIndexCounter[0] << '\t' << BinaryIndexCounter[1] << '\t' << BinaryIndexCounter[2] << '\t' << BinaryIndexCounter[3] << '\t' << BinaryIndexCounter[4] << '\t' << BinaryIndexCounter[5] << '\n';
    return Result;
}

int main(int ArgumentCount, char **ArgumentValue)
{
    FILE *TraceFilePointer;
    char *TraceFileName;
    float *Umatrix[6];
    float Filedata = 0.0;
    float *InputMatrixfromFile;
    float *InputSortedMatrixIndex;
    float *InputSortedMatrix;
    float *OutputMatrix;

    if (!(ArgumentCount == 2))
    {
        cout << "Error: Enter Input File" << ArgumentCount - 1 << '\n';
        exit(EXIT_FAILURE);
    }
    // cout << "A" << '\n';
    TraceFileName = ArgumentValue[1];
    TraceFilePointer = fopen(TraceFileName, "r");
    if (TraceFilePointer == NULL)
    {
        cout << "Error: Unable to open file " << TraceFileName << '\n';
        exit(EXIT_FAILURE);
    }
    InputMatrixfromFile = (float *)malloc(pow(2, 30) * sizeof(float));

    for (uint64_t RowCount = 0; RowCount < 6; RowCount++)
        Umatrix[RowCount] = (float *)malloc(4 * sizeof(float));

    // cout << "AB" << '\n';

    for (uint64_t RowCount = 0; RowCount < 6; RowCount++)
    {
        fscanf(TraceFilePointer, "%f %f", &Umatrix[RowCount][0], &Umatrix[RowCount][1]); // Line1
        fscanf(TraceFilePointer, "%f %f", &Umatrix[RowCount][2], &Umatrix[RowCount][3]); // Line2
    }

    while ((fscanf(TraceFilePointer, "%f", &Filedata)) == 1)
    {
        InputMatrixfromFile[InputMatrixSize] = Filedata;
        InputMatrixSize += 1;
    }
    fclose(TraceFilePointer);

    for (uint64_t RowCount = 0; RowCount < 6; RowCount++)
    {
        QbitPosition[5 - RowCount] = InputMatrixfromFile[InputMatrixSize - 1];
        InputMatrixSize = (InputMatrixSize - 1);
    }

    InputSortedMatrixIndex = (float *)malloc((InputMatrixSize) * sizeof(float));
    InputSortedMatrix = (float *)malloc((InputMatrixSize) * sizeof(float));
    OutputMatrix = (float *)malloc((InputMatrixSize) * sizeof(float));

    cudaError_t err = cudaSuccess;
    size_t InputMatrixFloatSize = InputMatrixSize * sizeof(float);
    size_t QuantumGateFloatSize = 6 * 4 * sizeof(float);
    // Allocate the host input vector A
    float *h_A = (float *)malloc(QuantumGateFloatSize);
    // Allocate the host input vector B
    float *h_B = (float *)malloc(InputMatrixFloatSize);
    // Allocate the host input vector C
    float *h_C = (float *)malloc(InputMatrixFloatSize);

    if ((h_A == NULL) || (h_B == NULL) || (h_C == NULL))
        cout << "Memory Allocate failed for host vectors" << '\n';

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, QuantumGateFloatSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, InputMatrixFloatSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, InputMatrixFloatSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /////////////////////////////////////////////////////Input Reordering///////////////////////////////////////////////////////
    for (uint64_t Index = 0; Index < InputMatrixSize; Index += pow(2, 6))
    {
        if (Index == 0)
        {
            for (uint64_t IndexIndex = 0; IndexIndex < pow(2, 6); IndexIndex++)
                InputSortedMatrixIndex[IndexIndex] = QuBitIndexCalculator(IndexIndex);
        }
        else
        {
            for (uint64_t IndexIndex = Index; IndexIndex < (Index + pow(2, 6)); IndexIndex++)
            {
                uint64_t IndexVariable = IndexIndex - pow(2, 6);
                InputSortedMatrixIndex[IndexIndex] = InputSortedMatrixIndex[IndexVariable] + 1;
            }

        CheckDuplicate:
            bool MatchFlag = false;
            for (uint64_t Search = 0; Search < Index; Search++)
            {
                if ((InputSortedMatrixIndex[Index]) == (InputSortedMatrixIndex[Search]))
                {
                    MatchFlag = true;
                    break;
                }
            }
            if (MatchFlag)
            {
                for (uint64_t IndexIndex = Index; IndexIndex < (Index + pow(2, 6)); IndexIndex++)
                    InputSortedMatrixIndex[IndexIndex] += 1;
                goto CheckDuplicate;
            }
        }
    }

    ////////////////////////////////////////////

    for (uint64_t QuantGateCounter = 0; QuantGateCounter < 6; QuantGateCounter++)
    {
        for (uint64_t Bsearch = 0; Bsearch < 4; Bsearch++)
        {
            h_A[(QuantGateCounter * 4) + Bsearch] = Umatrix[QuantGateCounter][Bsearch];
            // cout << "h_A[" << ((QuantGateCounter * 4) + Bsearch) << "]" << '\t' << h_A[(QuantGateCounter * 4) + Bsearch] << '\n';
        }
    }

    for (uint64_t Index = 0; Index < InputMatrixSize; Index++)
    {
        uint64_t ii = InputSortedMatrixIndex[Index];
        InputSortedMatrix[Index] = InputMatrixfromFile[ii];
        h_B[Index] = InputSortedMatrix[Index];
        // cout << "h_B[" << Index << "]" << '\t' << ii << '\t' << h_B[Index] << '\t' << InputSortedMatrix[Index] << '\n';
    }

    // Copy the host input vectors A in host memory to the device input vectors in device memory
    err = cudaMemcpy(d_A, h_A, QuantumGateFloatSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors B in host memory to the device input vectors in device memory
    err = cudaMemcpy(d_B, h_B, InputMatrixFloatSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // cout<<"Input Matrix Size:"<<InputMatrixSize<<'\n'<<'\n';

    dim3 BlockCount(InputMatrixSize / (32 * 2), 1);
    dim3 ThreadCount(16, 1);

    struct timeval begin, end;
    gettimeofday(&begin, NULL);
    QuantumGate<<<BlockCount, ThreadCount>>>(d_A, d_B, d_C, 32, InputMatrixSize / (32 * 2));
    gettimeofday(&end, NULL);
    uint64_t time_in_us = 1e6 * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec);
    // cout << "Run Time -> " << time_in_us << " us" << '\n';
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch QuantumGate kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    // printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, InputMatrixFloatSize, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    ///////////////////////////////////////////////////////////////////////////

    // for (uint64_t i = 0; i < InputMatrixSize; i++)
    // printf("%.3f\n", h_C[i]);
    // cout << "########################################" << '\n';

    for (uint64_t i = 0; i < (InputMatrixSize); i++)
    {
        uint64_t TempIndex = InputSortedMatrixIndex[i];
        OutputMatrix[TempIndex] = h_C[i];
    }

    ///////////////////////////////////////////////////////////////////////////
    // Free device global memory
    err = cudaFree(d_A);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (uint64_t i = 0; i < InputMatrixSize; i++)
        printf("%.3f\n", OutputMatrix[i]);

    return 0;
}
