#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

//Function declaration
static void matvec(int, int, double*, double*, double*, double*, double*);

int main(int argc, char** argv)
{
  //Initialize mpi
  MPI_Init(&argc,&argv);

  int n=80000;        //matrix dimension rows
  int m=5000;         //matrix dimension columns
  double *x,*b,*a,*p; //Pointer to input layer,biases, and result (activation)
  double* w;          //Pointer to weights matrix

  clock_t t;
  
  //Allocate memory for the arrays
  x = (double*)malloc(m*sizeof(double));
  a = (double*)malloc(n*sizeof(double));
  b = (double*)malloc(n*sizeof(double));
  p = (double*)malloc(n*sizeof(double));
  w = (double*)malloc(n*m*sizeof(double));

  
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
      w[i*m+j] = 1;


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
  free(w);
  free(p);

  MPI_Finalize();
  return 0; 
}


//Function definition
static void matvec(int n, int m, double* x, double* b, double* w, double* p, double* a)
{
  int rank,size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Status Status;
  
  double *locp, *loca, *locb, *locw;
  loca = (double*)malloc((n/size)*sizeof(double));
  locb = (double*)malloc((n/size)*sizeof(double));
  locp = (double*)malloc((n/size)*sizeof(double));
  locw = (double*)malloc((n*m/size)*sizeof(double));
  MPI_Barrier(MPI_COMM_WORLD);

  //Scatter data from 0 to other ranks
  MPI_Scatter(w, (n*m)/size, MPI_DOUBLE, locw, (n*m)/size, MPI_DOUBLE, 0, MPI_COMM_WORLD); //Scatter the weight matrix
  MPI_Scatter(b, n/size, MPI_DOUBLE, locb, n/size, MPI_DOUBLE, 0, MPI_COMM_WORLD); //Scatter the bias vector
  MPI_Scatter(p, n/size, MPI_DOUBLE, locp, n/size, MPI_DOUBLE, 0, MPI_COMM_WORLD); //Scatter the product holding vector)
  MPI_Bcast(x, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);// Broadcast the Vector
  
  //Product calculation loop
  for(int i=0;i<n/size;i++)
    for(int j=0;j<m;j++)
      locp[i] += locw[i*m+j]*x[j];
  
  for(int i=0;i<n/size;i++)
    loca[i] = locp[i]+locb[i];

  MPI_Gather(loca, n/size, MPI_DOUBLE, a, n/size, MPI_DOUBLE, 0, MPI_COMM_WORLD); // Gather the results
}
