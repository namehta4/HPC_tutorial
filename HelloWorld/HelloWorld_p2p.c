#include <stdio.h>
#include <string.h>
#include "mpi.h"

int main(int argv, char** argc)
{
  int rank, num, len;
  char hostname[MPI_MAX_PROCESSOR_NAME];
  
  MPI_Init(&argv,&argc);
  MPI_Comm_size(MPI_COMM_WORLD, &num);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name(hostname, &len);

  int tag=99;
  char message[20];
  MPI_Status status;

  if (rank==0)
  {
    strcpy(message, "Aloha Ao!");
    for(int i=1; i<num; i++)
    {
       MPI_Send(message, 13, MPI_CHAR, i, tag, MPI_COMM_WORLD);
    }
  }
  else
  {
    MPI_Recv(message, 20, MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);
  }
  printf("%.9s from rank %d \n",message,rank);

  MPI_Finalize();
  return 0;
}
