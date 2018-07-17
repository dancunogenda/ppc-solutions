#include<iostream>
#include<math.h>
#include "cp.h"
#include "vector.h"

#define VSIZE 8

using std::cout; 
using std::endl;
void print_mat(const float* m, int y, int x)
{
  cout<<"matrix: "<<endl;
  for(int j=0; j<y; j++)
  {
    for(int i=0; i<x; i++)
      cout<<m[j*x+i]<<" ";
    cout<<endl;
  }
}

void print_vec(const float8_t* m, int y, int x)
{
  cout<<"vector: "<<endl;
  for(int j=0; j<y; j++)
  {
    for(int i=0; i<x; i++)
      for(int k=0; k<VSIZE; k++)
        cout<<m[j*x+i][k]<<" ";
    cout<<endl;
  }
}
void correlate(int ny, int nx, const float* data, float* result)
{
  int vx= 0;
  int pad= nx%VSIZE;;
  if(pad)
    vx = (nx/VSIZE)+1;
  else
    vx = nx/VSIZE;  
  //float8_t normalised_data[vx*ny];
  float8_t* ndata = float8_alloc(vx*ny);

  for(int i=0; i<ny; i++)
    for(int j=0; j<nx; j++)
      ndata[(i*vx)+(j/VSIZE)][j%VSIZE]= data[i*nx+j];

  if(pad)
    for(int j=0; j<ny; j++)
      for(int i=pad; i<VSIZE; i++)
        ndata[j*vx+vx-1][i]=0.0;

  #pragma omp parallel for schedule(static,1)
  for (int y = 0; y < ny; ++y)
  {
    float mean = 0.0;
    float var = 0.0;
    float8_t mtmp = float8_0;
    for(int x=0; x<vx; x++)
      mtmp += ndata[x + y*vx];
    for(int i= 0;i<VSIZE;i++)
      mean += mtmp[i];
    mean = mean/nx;

    if(pad)
      for(int i=pad;i<VSIZE;i++)
        ndata[(vx-1)+y*vx][i]= mean;

    float8_t tmp=float8_0;
    float8_t vartmp=float8_0;
    for (int x=0; x<vx; x++)
    {
      tmp= ndata[x+y*vx]-mean;
      ndata[x+y*vx]= tmp;
      vartmp+= (tmp*tmp);
    }

    for(int i=0; i<VSIZE; i++)
      var+= vartmp[i];
    var= sqrt(var);
    for (int x= 0; x<vx; x++)
      ndata[y*vx +x] /= var;
  }
  #pragma omp parallel for schedule(static,1)
  for (int i=0; i<ny; i++)
  {
    for (int j=i; j<ny; j++)
    {
      float8_t sum= float8_0;
      double res = 0.0;
      for (int k=0; k<vx; k++)
        sum+= (ndata[i*vx+k] * ndata[j*vx+k]);
      for(int i =0;i<VSIZE;i++)
        res += sum[i];
      result[i*ny+j]= res;
    }
  }
}
