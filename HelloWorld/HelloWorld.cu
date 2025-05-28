#include<stdio.h>

//CUDA kernel 
__global__ void AlohaGPU() 
{
  printf("Aloha from GPU %d \n", threadIdx.x);
}

int main ()
{
  printf("Aloha from CPU \n");

  //Launch the CUDA kernel
  AlohaGPU<<1,5>>();
  cudaDeviceSynchronize();

  return 0;
}

