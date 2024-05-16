#include <stdio.h>
#include <time.h>

int main(int argc, char** argv)
{
  int n=3;            //matrix dimension rows
  int m=5;            //matrix dimension columns
  double* x,b,a;      //Pointer to input layer,biases, and result (activation)
  double** w;         //Pointer to weights matrix

  clock_t t;
  
  //Allocate memory for the arrays
  x = (double*)malloc(m*sizeof(double));
  b = (double*)malloc(n*sizeof(double));
  a = (double*)malloc(n*sizeof(double));
  w = (double**)malloc(n*sizeof(double*));
  for(int i=0;i<n;i++)
    w[i] = (double*)malloc(m*sizeof(double));

  
  //Initialize arrays
  for(int i=0;i<n;i++)
  {
    a[i] = 0.0;
    b[i] = 1.0;
  } 

  for(int j=0;j<m;j++)
    x[j] = 2.0;

  for(int i=0;i<n;i++)
    for(int j=0;j<m;j++)
      w[i][j] = 1;


  t = clock();
  //Call function for matrix-vector multiplication
  matvec(n,m,x,b,w,a);
  t = clock()-t;
  double time_taken = ((double)t)/CLOCKS_PER_SEC;
  printf("Time taken: %f \n",time_taken);


  //Free the memory utilized by the array
  free(x);
  free(b);
  free(a);
  for(int i=0;i<n;i++)
    free(w[i]);
  free(w);

  return 0; 
}


static void matvec(int n, int m, double* x, double* b, double** w, double* a)
{
  for(int i=0;i<n;i++)
    for(int j=0;j<m;j++)
      a[i] = w[i][j]*x[j] + b[i];
}
