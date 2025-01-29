#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "cuda_runtime.h"
#include "nccl.h"

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r != ncclSuccess) {                           \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// Function to initialize data
void initializeData(float* data, int size, float value) {
    for (int i = 0; i < size; i++) {
        data[i] = value;
    }
}

// Function to validate broadcasted data
bool validateData(float* data, int size, float expectedValue) {
    for (int i = 0; i < size; i++) {
        if (data[i] != expectedValue) {
            return false;
        }
    }
    return true;
}

int main() {
    int nDevices = 4;           // Number of GPUs
    int size = 1024 * 1024;     // Number of elements per buffer
    int root = 0;               // Root rank for broadcasting

    ncclComm_t comms[nDevices];
    float** sendbuff = (float**)malloc(nDevices * sizeof(float*));
    float** recvbuff = (float**)malloc(nDevices * sizeof(float*));
    cudaStream_t* streams = (cudaStream_t*)malloc(nDevices * sizeof(cudaStream_t));

    // Initialize CUDA and NCCL
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
        CUDACHECK(cudaStreamCreate(streams + i));
        
        // Initialize send buffer on the root rank with specific values
        if (i == root) {
            float* hostBuffer = (float*)malloc(size * sizeof(float));
            initializeData(hostBuffer, size, 123.45f); // Root rank initializes with value 123.45
            CUDACHECK(cudaMemcpy(sendbuff[i], hostBuffer, size * sizeof(float), cudaMemcpyHostToDevice));
            free(hostBuffer);
        }
    }

    NCCLCHECK(ncclCommInitAll(comms, nDevices, NULL));

    // Perform broadcast
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDevices; i++) {
        NCCLCHECK(ncclBroadcast((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, root, comms[i], streams[i]));
    }
    NCCLCHECK(ncclGroupEnd());

    // Synchronize and validate data on all devices
    bool valid = true;
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));

        // Copy receive buffer back to host and validate
        float* hostBuffer = (float*)malloc(size * sizeof(float));
        CUDACHECK(cudaMemcpy(hostBuffer, recvbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost));

        if (!validateData(hostBuffer, size, 123.45f)) {
            printf("Validation failed on device %d\n", i);
            valid = false;
        }

        free(hostBuffer);
    }

    // Output result of validation
    if (valid) {
        printf("Broadcast validation succeeded!\n");
    } else {
        printf("Broadcast validation failed!\n");
    }

    // Clean up resources
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
        NCCLCHECK(ncclCommDestroy(comms[i]));
    }

    free(sendbuff);
    free(recvbuff);
    free(streams);

    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

