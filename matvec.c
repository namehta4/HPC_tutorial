#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//Function declaration
static void matvec(int, int, double*, double*, double**, double*, double*);

int main(int argc, char** argv)
{
  int n=80000;        //matrix dimension rows
  int m=500;          //matrix dimension columns
  double *x,*b,*a,*p; //Pointer to input layer,biases, and result (activation)
  double** w;         //Pointer to weights matrix

  clock_t t;
  
  //Allocate memory for the arrays
  x = (double*)malloc(m*sizeof(double));
  a = (double*)malloc(n*sizeof(double));
  b = (double*)malloc(n*sizeof(double));
  p = (double*)malloc(n*sizeof(double));
  w = (double**)malloc(n*sizeof(double*));
  for(int i=0;i<n;i++)
    w[i] = (double*)malloc(m*sizeof(double));

  
  //Initialize arrays
  for(int i=0;i<n;i++)
  {
    a[i] = 0.0;
    b[i] = 1.0;
    p[i] = 0.0;
  } 

  for(int j=0;j<m;j++)
    x[j] = 2.0;

  for(int i=0;i<n;i++)
    for(int j=0;j<m;j++)
      w[i][j] = 1;


  t = clock();
  //Function call for matrix-vector multiplication
  for (int nstep=0;nstep<100;nstep++)
    matvec(n,m,x,b,w,p,a);
  t = clock()-t;
  double time_taken = ((double)t)/CLOCKS_PER_SEC;
  printf("Time taken: %f (sec) \n",time_taken);


  //Free the memory utilized by the array
  free(x);
  free(b);
  free(a);
  for(int i=0;i<n;i++)
    free(w[i]);
  free(w);
  free(p);

  return 0; 
}


//Function definition
static void matvec(int n, int m, double* x, double* b, double** w, double* p, double* a)
{
  for(int i=0;i<n;i++)
    for(int j=0;j<m;j++)
      p[i] += w[i][j]*x[j];
  
  for(int i=0;i<n;i++)
    a[i] = p[i]+b[i];
}
