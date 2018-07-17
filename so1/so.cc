#include "so.h"
#include <algorithm>
#include <iostream>
#include <omp.h>
using namespace std;
inline void merge(data_t A[], data_t B[], int m, int n) 
{
  int i=0, j=0, k=0;
  int size = m+n;
  data_t *C = (data_t *)malloc(size*sizeof(data_t));
  while (i < m && j < n)
  {
    if(A[i] < B[j])
      C[k++] = A[i++];
    else 
      C[k++] = B[j++];
  }
  if (i < m) 
  {
    for (int p = i; p < m;k++,p++ ) 
      C[k] = A[p];
  }
  if(j<n) 
  {
    for (int p = j; p < n; ) 
      C[k++] = B[p++];
  }
  for( i=0; i<size; i++ ) 
    A[i] = C[i];
  free(C);
}
 
void print(data_t *a, int size)
{
  for(int i=0; i<size; i++)
    cout<<a[i]<<" ";
  cout<<endl;
}

inline void arraymerge(data_t *a, int size, int *index, int N)
{
  int i;
  while ( N>1 ) 
  {
    for(i=0; i<N; i++ ) 
      index[i]=i*size/N; 
    index[N]=size;
    #pragma omp parallel for num_threads(i)
    for(i=0; i<N; i+=2 ) 
    {
      merge(a+index[i], a+index[i+1], 
          index[i+1]-index[i], index[i+2]-index[i+1]);
    }
    N=N>>1;
  }
}
 
void psort(int n, data_t* data) 
{
  int threads = omp_get_max_threads();
  if(threads==5 || threads== 3)
    threads--;
  if(threads==6 || threads==7)
    threads=4;
  if(threads>8)
    threads=8;
  int *index = (int *)malloc((threads+1)*sizeof(int));
  for(int i=0; i<threads; i++) 
  {
    index[i]=i*n/threads;
  }
  index[threads]=n;
  #pragma omp parallel for num_threads(threads)
  for(int i=0; i<threads; i++)
  {
    std::sort(data+index[i], data +index[i+1]);
  }
  if(threads>1 ) 
    arraymerge(data,n,index,threads);
}
