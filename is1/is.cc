#include "is.h"
#include "vector.h"
#include<iostream>
#include<omp.h>
using namespace std;

Result segment(int ny, int nx, const float* data) 
{
  double4_t *vdata= double4_alloc(ny*nx);
  double4_t Vpc = double4_0;
  #pragma omp parallel for
  for(int y=0; y<ny; y++)
  {
    for(int x=0; x<nx; x++)
    {
      vdata[y*nx+x]= double4_0;;
      vdata[y*nx+x][0]= data[3*((y*nx)+x)];
      vdata[y*nx+x][1]= data[3*((y*nx)+x)+1];
      vdata[y*nx+x][2]= data[3*((y*nx)+x)+2];
    }
  }

  //preprocessing of sum to make O(1) to find sum
  int sny= ny+1;
  int snx= nx+1;
  int sn= snx*sny;
  int N=nx*ny;
  double4_t *s= double4_alloc(sn);
  for(int y=0; y<sny; y++)
    s[y*snx]= double4_0;

  for(int x=0; x<snx; x++)
    s[x]= double4_0;

  //Not able to make this parallel
  for(int y=1; y<sny; y++)
  {
    for(int x=1; x<snx; x++)
    {
      s[y*(nx+1)+x]= s[(y-1)*(snx)+x] + 
                      s[y*(snx)+x-1] -
                      s[(y-1)*(snx)+x-1] +
                      vdata[(y-1)*nx+(x-1)];
      Vpc += vdata[(y-1)*nx+(x-1)];
    }
  }

  int optx0[8]={0,0,0,0,0,0,0,0};
  int opty0[8]={0,0,0,0,0,0,0,0};
  int optx1[8]={0,0,0,0,0,0,0,0};
  int opty1[8]={0,0,0,0,0,0,0,0};
  double opt[8]={0,0,0,0,0,0,0,0};
  //something fishy here lot of assembly instruction just to create this loop
  //not able to understand why these many instructions for this loop
  #pragma omp parallel for schedule(static, 1)
  for(int h=1; h<=ny; h++)
  {
    for(int w=1; w<=nx; w++)
    {
      double X=1.0/(h*w);
      double Y=0;
      if(N-(h*w))
        Y=1.0/(N-(h*w));
      double4_t divX= double4_0;
      divX[0]= X;
      divX[1]= X;
      divX[2]= X;
      double4_t divY= double4_0;
      divY[0]= Y;
      divY[1]= Y;
      divY[2]= Y;

      for(int y0=0; y0<ny-h+1; y0++)     
      {
        for(int x0=0; x0<nx-w+1; x0++)
        {
          int y1= y0+h;
          int x1= x0+w;
          double4_t Vxc = double4_0;
          Vxc= s[y1*snx+x1]+s[y0*snx+x0]-s[y0*snx+x1]-s[y1*snx+x0];
          double4_t Vyc= Vpc-Vxc;
          double4_t Fxy = (Vxc*Vxc*divX)+(Vyc*Vyc*divY);;
          double res=Fxy[0]+Fxy[1]+Fxy[2];

          int i=omp_get_thread_num();
          if( res > opt[i] )
          {
            optx0[i] = x0;
            opty0[i] = y0;
            optx1[i] = x1;
            opty1[i] = y1;  
            opt[i] = res;
            //asm("#dummy");
          }
        }
      }
    }
  }

  double max= 0.0;
  int index= 0;
  for(int i=0; i<8; i++)
  {
    if(opt[i]>max) {max= opt[i]; index= i;}
  }
  int opt_x0=optx0[index];
  int opt_y0=opty0[index];
  int opt_x1=optx1[index];
  int opt_y1=opty1[index];

  int X=(opt_x1-opt_x0)*(opt_y1-opt_y0);
  int Y=(ny*nx)-X;
  double4_t Astar= s[opt_y1*(nx+1)+opt_x1] +
                    s[opt_y0*(nx+1)+opt_x0] -
                    s[opt_y0*(nx+1)+opt_x1] -
                    s[opt_y1*(nx+1)+opt_x0];
  double4_t Bstar= double4_0;
  if(Y)
    Bstar= (Vpc-Astar)/Y;
  Astar/=X;

  float a[3];
  float b[3];
  a[0]=Astar[0]; a[1]=Astar[1]; a[2]=Astar[2];
  b[0]=Bstar[0]; b[1]=Bstar[1]; b[2]=Bstar[2];
  //for(int i=0; i<3; i++){ a[i]= Astar[i]; b[i]= Bstar[i];}
  Result result { opt_y0, opt_x0, opt_y1, opt_x1,
                  {b[0],b[1],b[2]},
                  {a[0],a[1],a[2]} };
  return result;
}
