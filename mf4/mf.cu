#include "mf.h"
#include<algorithm>
#include<iostream>

using std::cout;
using std::endl;

#define MIN(X, Y) ((X<Y)?X:Y)
#define MAX(X, Y) ((X>Y)?X:Y)
#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8

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

#define ELEM_SWAP(a,b) { register float t=(a);(a)=(b);(b)=t; }

__global__ void median(int ny, int nx, int hy, int hx, float *input, 
                        float *output, float* d_values )
{
  int x = threadIdx.x + (blockDim.x * blockIdx.x);
  int y = threadIdx.y + (blockDim.y * blockIdx.y);
  //float *values=new float[size*sizeof(float)];
  //__shared__ float values[size];

  //printf("\n x: %d y: %d",x,y);
  if(x>= nx || y>=ny)
    return;
  //int nx_blocks = nx/BLOCKDIM_X + (nx%BLOCKDIM_X == 0 ? 0:1);
  //float *values= d_values+(y*(BLOCKDIM_X*BLOCKDIM_Y)*nx_blocks + x *(BLOCKDIM_X+BLOCKDIM_Y));
  float *arr=(float *)malloc((2*hx+1)*(2*hy+1)*sizeof(float));

  int top= MAX(y-hy, 0);
  int bottom= MIN(y+hy, ny-1);

  int left= MAX(x-hx, 0);
  int right= MIN(x+hx, nx-1);
  //printf("\n top: %d bottom: %d left: %d right: %d",top, bottom, left, right);

  int k=0;
  for(int j= top; j<=bottom; j++)
    for(int i= left; i<=right; i++)
      arr[k++]= input[j*nx+i];

  //get the median of the arr
  int low, high ;
  int median;
  int middle, ll, hh;
  float med=0;

  low = 0 ; high = k-1 ; median = k/2;
  for (;;) 
  {
    if (high <= low) 
    {
      med= arr[median] ;
      break;
    }

    if (high == low + 1) 
    {  // Two elements only 
      if (arr[low] > arr[high])
        ELEM_SWAP(arr[low], arr[high]) ;
      med= arr[median] ;
      break;
    }

    // Find median of low, middle and high items; swap into position low 
    middle = (low + high) / 2;
    if (arr[middle] > arr[high])    ELEM_SWAP(arr[middle], arr[high]) ;
    if (arr[low] > arr[high])       ELEM_SWAP(arr[low], arr[high]) ;
    if (arr[middle] > arr[low])     ELEM_SWAP(arr[middle], arr[low]) ;

    // Swap low item (now in position middle) into position (low+1) 
    ELEM_SWAP(arr[middle], arr[low+1]) ;

    // Nibble from each end towards middle, swapping items when stuck
    ll = low + 1;
    hh = high;
    for (;;) 
    {
      do ll++; while (arr[low] > arr[ll]) ;
      do hh--; while (arr[hh]  > arr[low]) ;

      if (hh < ll)
      break;

      ELEM_SWAP(arr[ll], arr[hh]) ;
    }

    // Swap middle item (in position low) back into correct position
    ELEM_SWAP(arr[low], arr[hh]) ;

    // Re-set active partition
    if (hh <= median)
      low = ll;
      if (hh >= median)
      high = hh - 1;
  }

  if(k&1)
  {
    output[y*nx+x]= med;
  }
  else
  {
    int low, high ;
    int median;
    int middle, ll, hh;
    float med2=0;

    low = 0 ; high = k-1 ; median = k/2-1;
    for (;;) 
    {
      if (high <= low) 
      {
        med2= arr[median] ;
        break;
      }

      if (high == low + 1) 
      {  // Two elements only
        if (arr[low] > arr[high])
          ELEM_SWAP(arr[low], arr[high]) ;
        med2= arr[median] ;
        break;
      }

      // Find median of low, middle and high items; swap into position low
      middle = (low + high) / 2;
      if (arr[middle] > arr[high])    ELEM_SWAP(arr[middle], arr[high]) ;
      if (arr[low] > arr[high])       ELEM_SWAP(arr[low], arr[high]) ;
      if (arr[middle] > arr[low])     ELEM_SWAP(arr[middle], arr[low]) ;

      // Swap low item (now in position middle) into position (low+1)
      ELEM_SWAP(arr[middle], arr[low+1]) ;

      // Nibble from each end towards middle, swapping items when stuck
      ll = low + 1;
      hh = high;
      for (;;) 
      {
        do ll++; while (arr[low] > arr[ll]) ;
        do hh--; while (arr[hh]  > arr[low]) ;

        if (hh < ll)
        break;

        ELEM_SWAP(arr[ll], arr[hh]) ;
      }

      // Swap middle item (in position low) back into correct position
      ELEM_SWAP(arr[low], arr[hh]) ;

      // Re-set active partition 
      if (hh <= median)
        low = ll;
        if (hh >= median)
        high = hh - 1;
    }
    output[y*nx+x]= (med+med2)/2.0;
  }
  free(arr);
}

void printMatrix(const float *a, int ny, int nx) 
{
  for(int j=0; j<ny; j++)
  {
    for(int i=0; i<nx; i++)
    {   
      cout<<a[j*nx+i]<<" ";
    }   
    cout<<endl;
  }
}

void mf_cpu(int ny, int nx, int hy, int hx, const float* in, float* out) 
{
  int n=(2*hx+1)*(2*hy+1);
  float values[n];
  float temp;

  for(int y=0; y<ny; y++)
  {
    int top= MAX(y-hy, 0); 
    int bottom= MIN(y+hy, ny-1);
    for(int x=0; x<nx; x++)
    {   
      int left= MAX(x-hx, 0); 
      int right= MIN(x+hx, nx-1);
      int i= 0;
      for(int v=top; v<=bottom; v++)
        for(int u=left; u<=right;u++)
          values[i++]= in[v*nx +u];
      //best performance is with nth_element
      std::nth_element(values, values +(i/2), values+i);
      temp= values[i/2];
      if(i&1)
        out[y*nx +x]= temp;
      else
      {   
        std::nth_element(values, values +((i/2)-1), values+i);
        out[y*nx +x]= (temp+values[(i/2)-1])/2.0;
      }   
    }   
  }
}

void mf(int ny, int nx, int hy, int hx, const float* in, float* out) 
{
  //cout<<"nx: "<<nx<<" ny: "<<ny<<" hx: "<<hx<<" hy: "<<hy<<endl;
  int size= (nx*ny*sizeof(float));
  float *d_in= 0;
  float *d_out= 0;
  float *d_values=0;
  float v_size= ((2*hx+1)*(2*hy+1))*(BLOCKDIM_X*BLOCKDIM_Y);
  //CHECK_CUDA_ERROR(cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice));

  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
  int nx_blocks = nx/BLOCKDIM_X + (nx%BLOCKDIM_X == 0 ? 0:1);
  int ny_blocks = ny/BLOCKDIM_Y + (ny%BLOCKDIM_Y == 0 ? 0:1);
  dim3 grid( nx_blocks, ny_blocks);
  //cout<<"nx_blocks: "<<nx_blocks<<" ny_blocks: "<<ny_blocks<<endl;
  //dim3 threads(ny, nx);
  //dim3 grid( ny, nx);
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_in, size));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_out, size));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_values, v_size));
  CHECK_CUDA_ERROR(cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice));

  median<<<grid, threads>>>(ny, nx, hy, hx, d_in, d_out, d_values);

  CHECK_CUDA_ERROR(cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost));

  cudaFree(d_in);
  cudaFree(d_out);
  //cudaFree(d_values);

  //cout<<"in: "<<endl;
  //printMatrix(in, ny, nx);
  //cout<<"out: "<<endl;
  //printMatrix(out, ny, nx);
}
