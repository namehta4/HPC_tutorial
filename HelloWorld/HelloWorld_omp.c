#include <stdio.h>
#include "omp.h"

int main(int argv, char** argc)
{
  omp_set_num_threads(16);
  #pragma omp parallel
  {
    int id=omp_get_thread_num();
    printf("Aloha(%d) Ao(%d)! \n",id,id);
  }
  return 0;
}

