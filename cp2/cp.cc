#include<iostream>
#include<numeric>
#include<algorithm>
#include<functional>
#include "cp.h"

void correlate(int ny, int nx, const float* data, float* result) 
{
  //float normalised_data[ny*nx];
  float *ndata= (float *)malloc(sizeof(float)*ny*nx);
  #pragma omp parallel for 
  for(int i=0; i<ny; i++)
  {
    int s= i*nx;
    float mean=0.0f;
    for(int i=0; i<nx; i++)
      mean+= data[s+i];
    mean/=nx;
    float var=0.0f;
    float tmp= 0.0;
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
  //#pragma omp parallel for schedule(dynamic)
  #pragma omp parallel for schedule(static, 1)
  for(int j=0; j<ny; j++)
  {
    for(int i=j; i<ny; i++)
    {
      float sum=0.0f;
      for(int k=0; k<nx; k++)
        sum+= (ndata[j*nx+k]*ndata[i*nx+k]);
      //why cant i do this
      //#pragma omp atomic
      result[i+ny*j]= sum;
    }
  }
}
