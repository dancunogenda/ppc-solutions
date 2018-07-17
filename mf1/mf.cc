#include "mf.h"
#include<algorithm>

#define MIN(X, Y) ((X<Y)?X:Y)
#define MAX(X, Y) ((X>Y)?X:Y)

#define ELEM_SWAP(a,b) { register float t=(a);(a)=(b);(b)=t; }

float kth_smallest(float a[], int n, int k)
{
    register int i,j,l,m ;
    register float x ; 

    l=0 ; m=n-1 ;
    while (l<m) {
        x=a[k] ;
        i=l ;
        j=m ;
        do {
            while (a[i]<x) i++ ;
            while (x<a[j]) j-- ;
            if (i<=j) {
                ELEM_SWAP(a[i],a[j]) ;
                i++ ; j-- ;
            }   
        } while (i<=j) ;
        if (j<k) l=i ;
        if (k<i) m=j ;
    }   
    return a[k] ;
}

#define median(a,n) kth_smallest(a,n,(n)/2)

float quick_select(float arr[], int n, int k) 
{
    int low, high ;
    int median;
    int middle, ll, hh;

    low = 0 ; high = n-1 ; median = k;
    for (;;) {
        if (high <= low) /* One element only */
            return arr[median] ;

        if (high == low + 1) {  /* Two elements only */
            if (arr[low] > arr[high])
                ELEM_SWAP(arr[low], arr[high]) ;
            return arr[median] ;
        }

    /* Find median of low, middle and high items; swap into position low */
    middle = (low + high) / 2;
    if (arr[middle] > arr[high])    ELEM_SWAP(arr[middle], arr[high]) ;
    if (arr[low] > arr[high])       ELEM_SWAP(arr[low], arr[high]) ;
    if (arr[middle] > arr[low])     ELEM_SWAP(arr[middle], arr[low]) ;

    /* Swap low item (now in position middle) into position (low+1) */
    ELEM_SWAP(arr[middle], arr[low+1]) ;

    /* Nibble from each end towards middle, swapping items when stuck */
    ll = low + 1;
    hh = high;
    for (;;) {
        do ll++; while (arr[low] > arr[ll]) ;
        do hh--; while (arr[hh]  > arr[low]) ;

        if (hh < ll)
        break;

        ELEM_SWAP(arr[ll], arr[hh]) ;
    }

    /* Swap middle item (in position low) back into correct position */
    ELEM_SWAP(arr[low], arr[hh]) ;

    /* Re-set active partition */
    if (hh <= median)
        low = ll;
        if (hh >= median)
        high = hh - 1;
    }
}

void mf(int ny, int nx, int hy, int hx, const float* in, float* out) 
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
      /*
      //almost equal to nth_element
      temp= quick_select(values, i, i/2);
      if(i&1)
        out[y*nx + x]= temp;
      else
        out[y*nx + x]= (temp + quick_select(values, i, ((i/2)-1)))/2.0;
      */
      /*
      //worst performance
      std::sort(values, values+i);      
      if(i&1)
        out[y*nx + x]= (values[i/2]);
      else
        out[y*nx + x]= ((values[i/2] + values[i/2 -1])/2.0);
      */
    }   
  }
}
