#include "mf.h"
#include<algorithm>
#include<vector>

using std::vector;
#define MIN(X, Y) ((X<Y)?X:Y)
#define MAX(X, Y) ((X>Y)?X:Y)

void mf(int ny, int nx, int hy, int hx, const float* in, float* out) 
{
  vector<float> values;
  //int n=(2*hx+1)*(2*hy+1);
  for(int y=0; y<ny; y++)
  {
    int top= MAX(y-hy, 0); 
    int bottom= MIN(y+hy, ny-1);
    float median;
    int n;
    for(int x=0; x<nx; x++)
    {   
      int left= MAX(x-hx, 0); 
      int right= MIN(x+hx, nx-1);
      values.clear();
      for(int v=top; v<=bottom; v++)
        for(int u=left; u<=right;u++)
          values.push_back(in[v*nx +u]);
      //best performance is with nth_element
      n= values.size();
      std::nth_element(values.begin(), values.begin() +(n/2), values.end());
      median= values[n/2];
      if(n&1)
        out[y*nx +x]= median;
      else
      {
        std::nth_element(values.begin(), values.begin()+(n/2-1), values.end());
        out[y*nx +x]= (median+values[n/2-1])/2.0;
      }
    }   
  }
}
