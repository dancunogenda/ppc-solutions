#include<iostream>
#include<numeric>
#include<algorithm>
#include<functional>
#include "cp.h"

void correlate(int ny, int nx, const float* data, float* result) 
{
  //float normalised_data[ny*nx];
  float *ndata= (float *)malloc(sizeof(float)*ny*nx);
  //normalise the matrix
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
  //find correlation
  for(int j=0; j<ny; j++)
  {
    for(int i=j; i<ny; i++)
    {
      float sum=0.0f;
      for(int k=0; k<nx; k++)
        sum+= (ndata[j*nx+k]*ndata[i*nx+k]);
      result[i+ny*j]= sum;
    }
  }
}
