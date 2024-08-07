#include <stdio.h>
#include "mpi.h"

int main(int argv, char** argc)
{
  int rank, num, len;
  char hostname[MPI_MAX_PROCESSOR_NAME];
  
  MPI_Init(&argv,&argc);
  MPI_Comm_size(MPI_COMM_WORLD, &num);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name(hostname, &len);

  printf("On host %s\n", hostname);
  printf("Aloha Ao! from %d/%d\n",rank,num);

  MPI_Finalize();
  return 0;
}
