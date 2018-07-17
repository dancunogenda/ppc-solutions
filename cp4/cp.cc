//----------------------------------------------------------------------------
// FILE: cp.cc
// DESCRIPTION: cp4 impplementation with instruction serialising and vectors
// AUTHOR: Suhas Thejaswi
// DATE: 25-AUG-2015
//----------------------------------------------------------------------------

#include<iostream>
#include<math.h>
#include "cp.h"
#include "vector.h"

#define VSIZE 8
#define BSIZE 6
#define INSLEN 8
#define MIN(X,Y) (X)<(Y)?X:Y
#define MAX(X,Y) (X)>(Y)?X:Y
#define NOREDUNDANT 1

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

  int blkpad=0;
  while(vx%INSLEN)
  {
    vx++;
    blkpad++;
  }
  //cout<<"vx: "<<vx<<endl;
  float8_t* ndata = float8_alloc((vx)*ny);
  
  //copy the matrix to vector
  for(int i=0; i<ny; i++)
    for(int j=0; j<nx; j++)
      ndata[(i*vx)+(j/VSIZE)][j%VSIZE]= data[i*nx+j];
  //initialise padded elements to 0
  if(pad)
    for(int j=0; j<ny; j++)
      for(int i=pad; i<VSIZE; i++)
        ndata[j*vx+vx-1-blkpad][i]=0.0;

  //padding to make blocks of length 8
  if(blkpad)
    for(int j=0; j<ny; j++)
      for(int i=vx-blkpad; i<vx; i++)
        ndata[j*vx+i]= float8_0;
  //print_mat(data, ny, nx);
  //print_vec(ndata, ny, vx);
  //Normalise the matrix
  #pragma omp parallel for
  for (int y = 0; y < ny; ++y)
  {
    float mean = 0.0;
    float var = 0.0;
    float8_t mtmp = float8_0;
    //calculate mean
    for(int x=0; x<vx; x++)
      mtmp += ndata[x + y*vx];
    for(int i= 0;i<VSIZE;i++)
      mean += mtmp[i];
    mean = mean/nx;

    if(pad)
      for(int i=pad;i<VSIZE;i++)
        ndata[(vx-1-blkpad)+y*vx][i]= mean;

    const float8_t float8_mean= {mean,mean,mean,mean,mean,mean,mean,mean};
    if(blkpad)
      for(int i=vx-blkpad; i<vx; i++)
        ndata[y*vx+i]= float8_mean;

    float8_t tmp=float8_0;
    float8_t vartmp=float8_0;
    //calculate varience
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
  //print_vec(ndata, ny, vx);

  //matrix multiplication
  #pragma omp parallel for schedule(static,1)
  for (int j=0; j<ny; j+=BSIZE)
  {
    int maxu= MIN(j+BSIZE, ny);
    for (int i=j; i<ny; i+=BSIZE)
    {
      int maxv= MIN(i+BSIZE, ny);
      for(int u=j; u<maxu; u++)
      {
        for(int v=i; v<maxv ; v++)
        {
          //BUG-BUG not sure how open-mp handling of IF-clause within loop
          //Performance is same even without this if-clause
          //But there are some redundant calculations without the if-clause
          //I am not sure why performance is not changing
          #ifdef NOREDUNDANT
          if(v>=u)
          {
          #endif
            //cout<<"u: "<<u<<" v: "<<v<<endl;
            float8_t sum_0= float8_0;
            float8_t sum_1= float8_0;
            float8_t sum_2= float8_0;
            float8_t sum_3= float8_0;
            float8_t sum_4= float8_0;
            float8_t sum_5= float8_0;
            float8_t sum_6= float8_0;
            float8_t sum_7= float8_0;
            for(int k=0; k<vx; k+=INSLEN)
            {
              sum_0+= (ndata[u*vx+k] * ndata[v*vx+k]);
              sum_1+= (ndata[u*vx+(k+1)] * ndata[v*vx+(k+1)]);
              sum_2+= (ndata[u*vx+(k+2)] * ndata[v*vx+(k+2)]);
              sum_3+= (ndata[u*vx+(k+3)] * ndata[v*vx+(k+3)]);
              sum_4+= (ndata[u*vx+(k+4)] * ndata[v*vx+(k+4)]);
              sum_5+= (ndata[u*vx+(k+5)] * ndata[v*vx+(k+5)]);
              sum_6+= (ndata[u*vx+(k+6)] * ndata[v*vx+(k+6)]);
              sum_7+= (ndata[u*vx+(k+7)] * ndata[v*vx+(k+7)]);
            }
            float8_t sum= sum_0+sum_1+sum_2+sum_3+sum_4+sum_5+sum_6+sum_7;
            float res = 0.0f;
            for(int z =0;z<VSIZE;z++)
              res += sum[z];
            result[u*ny+v]= res;
          #ifdef NOREDUNDANT
          }
          #endif
        }
      }
    }
  }
  //print_mat(result, ny,ny);
}
