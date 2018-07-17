#include "mf.h"
#include<algorithm>

#define MIN(X, Y) ((X<Y)?X:Y)
#define MAX(X, Y) ((X>Y)?X:Y)

void mf(int ny, int nx, int hy, int hx, const float* in, float* out) 
{
  int n=(2*hx+1)*(2*hy+1);
  #pragma omp parallel for
  for(int y=0; y<ny; y++)
  {
    int top= MAX(y-hy, 0); 
    int bottom= MIN(y+hy, ny-1);
    float temp;
    float values[n];
    #pragma omp parallel for
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
