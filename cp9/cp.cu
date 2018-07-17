#include "cp.h"
#include <math.h>
#include <cuda_runtime.h>
#include<iostream>
#include<iomanip>

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16
#define TILE_WIDTH 16
#define TILE_WIDTH_X 16
#define TILE_WIDTH_Y 16

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

__global__ void matrixMultiply(float * A, float* B, float * C, int ny, int nx)
{
  __shared__ float ds_M[BLOCKDIM_X][TILE_WIDTH];
  __shared__ float ds_N[BLOCKDIM_Y][TILE_WIDTH];
  int bx= blockIdx.x; 
  int by= blockIdx.y;
  int tx= threadIdx.x; 
  int ty= threadIdx.y;
  int row= by * blockDim.y + ty;
  int col= bx * blockDim.x + tx;
  float v= 0.0f;

  //calculate the boundaries of block
  //this saved me lot of time
  if(blockIdx.x < blockIdx.y)
    return;
  //calculating this outside takes more time apparantly
  //int n_tiles = nx/TILE_WIDTH + (nx%TILE_WIDTH == 0 ? 0:1);
  //for (int m= 0; m<n_tiles; ++m) 
  for (int m= 0; m<(nx-1)/TILE_WIDTH+1; ++m) 
  {
    if(row < ny && m*TILE_WIDTH+tx < nx)
      ds_M[ty][tx] = A[row*nx + m*TILE_WIDTH+tx];
    else
      ds_M[ty][tx] = 0;

    if(col < ny && m*TILE_WIDTH+ty < nx)
      ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*ny+col];
      //ds_N[ty][tx] = A[col*nx + m*TILE_WIDTH+ty];
    else
      ds_N[ty][tx] = 0;

    __syncthreads();

    //calculating only upper half of the matrix. to exploit symmetry
    //Is this a good way to do it.. damn this works only if entire warp has (col<row)
    if(col<row)
      continue;
    for (int k = 0; k < TILE_WIDTH; k++)
      v+= ds_M[ty][k] * ds_N[k][tx];

    __syncthreads();
  }
  if (row < ny && col < ny)
    C[row*ny+col]= v;
}

void printMatrix(const float *a, int ny, int nx) 
{
  //cout<<endl;
  for(int j=0; j<ny; j++)
  {
    for(int i=0; i<nx; i++)
    {   
      //cout<<setw(10)<<a[j*nx+i];
      printf("%f ",a[j*nx+i]);
    }   
    //cout<<endl;
    printf("\n");
  }
}
 
void correlate(int ny, int nx, const float* data, float* result) 
{
  float* ndata = (float*)malloc(ny*nx*sizeof(float));
  float* ndata_t = (float*)malloc(ny*nx*sizeof(float));

  for(int j=0; j<ny; j++)
  {
    int s= j*nx;
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
    {
      ndata[s+i]/=var;
      ndata_t[i*ny+j]= ndata[s+i];
    }
  }

  /*
  for(int j=0; j<ny; j++)
    for(int i=0; i<nx; i++)
      ndata_t[i*ny+j]=ndata[j*nx+i];
  */

  //matrix multiplication
  float *d_data;
  float *d_data_t;
  float *d_res;
  int size = nx*ny*sizeof(float);
  CHECK_CUDA_ERROR(cudaMalloc((void**) &d_data, size));
  CHECK_CUDA_ERROR(cudaMalloc((void**) &d_data_t, size));
  CHECK_CUDA_ERROR(cudaMalloc((void**) &d_res, ny*ny*sizeof(float)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_data, ndata, size,cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_data_t, ndata_t, size,cudaMemcpyHostToDevice));
  int nx_blocks = ny/BLOCKDIM_X + (ny%BLOCKDIM_X == 0 ? 0:1);
  int ny_blocks = ny/BLOCKDIM_Y + (ny%BLOCKDIM_Y == 0 ? 0:1);
  //int nx_blocks = ny/TILE_WIDTH_X + (ny%TILE_WIDTH_X == 0 ? 0:1);
  //int ny_blocks = ny/TILE_WIDTH_Y + (ny%TILE_WIDTH_Y == 0 ? 0:1);
  dim3 grid(nx_blocks, ny_blocks);
  //dim3 block(TILE_WIDTH_X,TILE_WIDTH_Y);
  dim3 block(BLOCKDIM_X,BLOCKDIM_Y);

  //printMatrix(ndata, ny, nx);
  matrixMultiply<<< grid, block >>>(d_data, d_data_t, d_res, ny, nx);
                                 
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaMemcpy(result, d_res, ny*ny*sizeof(float), 
        cudaMemcpyDeviceToHost));

  //for(int i=0; i<ny; i++)
    //result[i*ny+i]=1.0f;
  //printMatrix(result, ny, ny);
  free(ndata);
  free(ndata_t);
  cudaFree(d_data);
  cudaFree(d_data_t);
  cudaFree(d_res);
}
