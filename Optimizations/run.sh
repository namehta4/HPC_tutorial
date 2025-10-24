#!/bin/bash

nvcc -O0 -G matvec_opt0.cu 
echo "*****************"
for (( i = 1; i <= 10; i++ ))
do 
  srun -n 1 a.out
done
srun -n 1 a.out
srun -n 1 nsys profile --stats=true -t cuda a.out


nvcc -O0 -G matvec_opt1.cu 
echo "*****************"
for (( i = 1; i <= 10; i++ ))
do 
  srun -n 1 a.out
done
srun -n 1 a.out
srun -n 1 nsys profile --stats=true -t cuda a.out


nvcc -O0 -G matvec_opt2.cu 
echo "*****************"
for (( i = 1; i <= 10; i++ ))
do 
  srun -n 1 a.out
done
srun -n 1 a.out
srun -n 1 nsys profile --stats=true -t cuda a.out


nvcc -O0 -G matvec_opt3.cu 
echo "*****************"
for (( i = 1; i <= 10; i++ ))
do 
  srun -n 1 a.out
done
srun -n 1 a.out
srun -n 1 nsys profile --stats=true -t cuda a.out
