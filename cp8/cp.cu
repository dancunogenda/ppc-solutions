#include "cp.h"
#include <math.h>
#include <cuda_runtime.h>
#include<iostream>
using namespace std;
#define BLOCK_SIZE 8

//#define GPU_NORMALISATION 1

#define CHECK_CUDA_ERROR(call) \
        do { \
          cudaError_t result_ = (call); \
          if (result_ != cudaSuccess) \
          { \
            fprintf(stderr, #call " failed: %s\n", \
                    cudaGetErrorString(result_)); \
            exit(1); \
          } \
        } while(0)
__global__ void var(float *input,float *output, int N, float mean)
{
  int idx=threadIdx.x+(blockDim.x*blockIdx.x);
  if (idx < N) output[idx] = (input[idx]-mean)*(input[idx]-mean);
}

__global__ void norm(float *input, int N,float mean,float sd)
{
  int idx=threadIdx.x+(blockDim.x*blockIdx.x);
  if (idx < N) input[idx] =  (input[idx]-mean)/sd;
}
 
__global__ void matrixMul( float* C, float* A, int ny,int nx)
{
   int tx = threadIdx.x + (blockDim.x * blockIdx.x);
   int ty = threadIdx.y + (blockDim.y * blockIdx.y);
  
   if(tx>= ny || ty>=ny)
    return;
   float value = 0;
   if(tx<ty)
    return;
   for (int i = 0; i < nx; ++i)
   {
      float elementA = A[ty * nx + i];
      float elementB = A[tx * nx + i];
      value += elementA * elementB;
   }
    C[ty * ny + tx] = value;
}
 
void correlate(int ny, int nx, const float* data, float* result) 
{
  float* ndata = (float*)malloc(ny*nx*sizeof(float));

  #ifdef GPU_NORMALISATION
  float *a_d_input;
  float *a_d_output; 
  float *a_h= (float*)malloc(nx*sizeof(float));
  size_t rsize = nx * sizeof(float);
  CHECK_CUDA_ERROR(cudaMalloc((void **) &a_d_input, rsize)); 
  CHECK_CUDA_ERROR(cudaMalloc((void **) &a_d_output, rsize)); 
  //normalise the matrix
  for (int y = 0; y < ny; ++y) 
  {
    float mean = 0.0;
    float sd = 0.0;

    //Finding the mean
    for (int x = 0; x < nx; ++x) 
    {
      ndata[x + y*nx] = data[x + y*nx];
      mean += ndata[x + y*nx];
    }
    mean= mean/nx;
    CHECK_CUDA_ERROR(cudaMemcpy(a_d_input, &ndata[y*nx], rsize, 
          cudaMemcpyHostToDevice));
    int block_size = 10;
    int n_blocks = nx/block_size + (nx%block_size == 0 ? 0:1);
    var<<< n_blocks, block_size >>> (a_d_input,a_d_output,nx,mean);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaMemcpy(a_h, a_d_output, rsize,
          cudaMemcpyDeviceToHost));
    for (int x= 0; x< nx; ++x) 
      sd += a_h[x];

    sd= sqrt(sd);
    CHECK_CUDA_ERROR(cudaMemcpy(a_d_output, &ndata[y*nx], rsize, 
          cudaMemcpyHostToDevice));
    norm<<< n_blocks, block_size >>> (a_d_output,nx,mean,sd);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaMemcpy(&ndata[y*nx], a_d_output, rsize, 
          cudaMemcpyDeviceToHost));
  }
  free(a_h);
  cudaFree(a_d_input);
  cudaFree(a_d_output);
  #endif

  #ifndef GPU_NORMALISATION
  for(int i=0; i<ny; i++)
  {
    int s= i*nx;
    float mean=0.0f;
    for(int i=0; i<nx; i++)
    {
      mean+= data[s+i];
    }
    mean/=nx;
    float var=0.0f;
    float tmp= 0.0f;
    for(int i=0; i<nx; i++)
    {   
      tmp= data[s+i]-mean;
      ndata[s+i]= tmp;
      var+= tmp*tmp;
    }   
    var=std::sqrt(var);
    for(int i=0; i<nx; i++)
      ndata[s+i]/=var;
  }
  #endif

  //matrix multiplication
  float *d_data;
  float *d_res;
  int size = nx*ny*sizeof(float);
  CHECK_CUDA_ERROR(cudaMalloc((void**) &d_data, size));
  CHECK_CUDA_ERROR(cudaMalloc((void**) &d_res, ny*ny*sizeof(float)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_data, ndata, size,cudaMemcpyHostToDevice));
  dim3 threads(BLOCK_SIZE,BLOCK_SIZE);
  int n_blocks = ny/BLOCK_SIZE + (ny%BLOCK_SIZE == 0 ? 0:1);
  dim3 grid(n_blocks,n_blocks);

  matrixMul<<< grid, threads >>>(d_res, d_data, ny,nx);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaMemcpy(result, d_res, ny*ny*sizeof(float), 
        cudaMemcpyDeviceToHost));

  free(ndata);
  cudaFree(d_data);
  cudaFree(d_res);
}
