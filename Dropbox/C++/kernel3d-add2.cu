#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <unistd.h> 
#include <sys/time.h>

__global__ void add(float *array, int dimensions, int num_elements)
{
  int index = (threadIdx.x + blockIdx.x * blockDim.x) * dimensions;
  // + blockIdx.y * blockDim.y + blockIdx.z * blockDim.z;
  
  if (threadIdx.x + blockIdx.x * blockDim.x < num_elements) {
    for (int d = 0; d < dimensions; d++) {
      array[index + d] = index + d + 1;
    }
  }
}

__global__ void makezero(float * __restrict__ array, int dimensions, int num_elements)
{
  int index = (threadIdx.x + blockIdx.x * blockDim.x) * dimensions;
  //int index = threadIdx.x + threadIdx.y + threadIdx.z + blockIdx.x * blockDim.x + blockIdx.y * blockDim.y + blockIdx.z * blockDim.z;
  if (threadIdx.x + blockIdx.x * blockDim.x < num_elements) {
    for (int d = 0; d < dimensions; d++)
    {
      array[index + d] = 0;
    }
  }
}



//works, don't change
__global__ void distance(float * __restrict__ solution, float * __restrict__ array, int dimensions, int size)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  float dist;
  if (index < size) {
    for (int i = 0; i < size; i++) {
      solution[index*size + i] = 0;
      for (int d = 0; d < dimensions; d++) {
        dist = array[i*dimensions + d] - array[index * dimensions + d];
        solution[index*size + i] += dist * dist;
      }
      solution[index*size + i] = sqrt(solution[index*size + i]);
    }
  }
}



//works, don't change
__global__ void distance2(float * __restrict__ solution, float * __restrict__ array, int dimensions, int size)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int i = index/size + index;
  float dist;
  while (i < size) {
    solution[index * size + i] = 0;
    for (int d = 0; d < dimensions; d++) {
      dist = (array[i * dimensions + d] - array[index * dimensions + d]);
      solution[index * size + i] +=  dist * dist;
    }
    solution[index*size + i] = sqrt(solution[index*size + i]);
    i++;
  }
}


//works, don't change
__global__ void distance3(float * __restrict__ solution, float * __restrict__ array, int dimensions, int x_size)
{    
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < (x_size * (x_size - 1)) / 2) {
    float dist;
    int column;
    int row;
    int rev_index = (x_size * (x_size - 1)) / 2 - index - 1;
    x_size += -1;
    column = - ((-1 + sqrt(double (1 + 8 * rev_index))) / 2);
    column = column + x_size - 1;
    row = index + column + 1 - column * x_size + (column * (column - 1)) / 2;
   
    solution[index] = 0;
    for (int d = 0; d < dimensions; d++) {
      dist = (array[column * dimensions + d] - array[row * dimensions + d]);
      solution[index] +=  dist * dist;
    }
    solution[index] = sqrt(solution[index]);
  }
}

float array_sum(float * array, int array_length)
{
  int i = 0;
  float array_sum = 0;
  for (i = 0; i < array_length; i++) {
    array_sum += array[i];
  }
  return array_sum;
}

int check_solution(float * array_solution, int array_size, float test_value)
{
  float array_summation = 0;
  array_summation = array_sum(array_solution, array_size);
  if (array_summation == test_value) {
    printf("Array solution is correct\n");
  } else {
    printf("Array solution is NOT correct: %f \n",array_summation);
  }
  return 0;
}

void print_cuda_error()
{
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("\nCUDA error: %s\n", cudaGetErrorString(error));
  } else {
      printf("\nno CUDA errors, all good\n");
  }
}

float * centroid(float * array, int array_length, int dimensions)
{
  float * average;
  average = new float [dimensions];
  for (int i = 0; i < array_length; i += dimensions) {
    for (int d = 0; d < dimensions; d ++) {
      average[d] += array[i + d];
    }
  }
  for (int d = 0; d < dimensions; d ++) {
    average[d] = average[d] / (array_length / dimensions);
    //printf ("%f", average[d]);
  }
  return average;
}

__global__ void move(float * __restrict__ array, float * vector, int dimensions, int x_size)
{    
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < x_size) {
    for (int d = 0; d < dimensions; d++) {
      array[index + d] += vector[d];
    }
  }
}


int main(void)
{ 
  cudaEvent_t startEvent, stopEvent;
  float ms = 0;
  int num_elements = 5;
  int dimensions = 3;
  int repeats = 1000;

  //int num_bytes = num_elements * dimensions * sizeof(float);
  struct timeval tv1, tv2;
  
  float *device_array = 0;
  float *host_array = 0;
  float *host_solution = 0;
  float *device_solution = 0;
  // allocate memory for CPU
  host_array = (float*)malloc(num_elements * dimensions * sizeof(float)); //array on the computer
  host_solution = (float*)malloc(num_elements * num_elements * sizeof(float)); //distance matrix on the computer
  // allocate memory for GPU
  cudaMalloc((void**)&device_array, num_elements * dimensions * sizeof(float));  //GPU array
  cudaMalloc((void**)&device_solution, num_elements * num_elements * sizeof(float)); // GPU distance matrix
  
  // test if solution was correctly initialized
  check_solution(host_solution, num_elements*num_elements, 0);
  // create grid
  int max_gridsize = 1024; 
  dim3 grid_size; // two dimensional grid
  grid_size.y = 1;
  grid_size.z = 1;
  if (num_elements < max_gridsize) {
    grid_size.x = num_elements;
  } else {
    grid_size.x = max_gridsize;
  }
  
  // create blocks
  int blocks = (num_elements + grid_size.x - 1)/grid_size.x;
  dim3 block_size;
  block_size.x = blocks;
  block_size.y = 1;
  block_size.z = 1;
  
  // print CUDA variables
  printf("Blocks: %i \n",block_size.x);
  printf("Grid size: %i \n",grid_size.x);
  
  
  printf("Number of elements: %i \n", num_elements);
  printf("Dimensions: %i \n", dimensions);
    
  printf("\n###########\n");
  printf("\nStarted\n");
  printf("\n###########\n");
  makezero<<<grid_size.x, blocks>>>(device_array, dimensions, num_elements);
  cudaMemcpy(host_array, device_array, num_elements * dimensions * sizeof(float), cudaMemcpyDeviceToHost);
  add<<<grid_size.x, blocks>>>(device_array, dimensions, num_elements);
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);

  cudaMemcpy(host_array, device_array, num_elements * dimensions * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(device_solution, host_solution, num_elements * num_elements * sizeof(float), cudaMemcpyHostToDevice);
  cudaEventRecord(startEvent,0);
  //gettimeofday(&tv1, NULL);
  
  //distance<<<1,1024>>>(device_solution, device_array, dimensions, num_elements);
  //distance<<<1,1>>>(device_solution, device_array, dimensions, num_elements);
  //cudaMemcpy(host_solution, device_solution, num_elements * num_elements * sizeof(float), cudaMemcpyDeviceToHost);
  

  printf ("%i blocks\n",blocks);
  printf ("%i grid_size.x\n",grid_size.x);
 
////////////////////////////////////////////////////////////////////////////////////////    
  gettimeofday(&tv1, NULL);
  for (int i = 0; i < repeats; i++) {
    distance<<<blocks,grid_size.x>>>(device_solution, device_array, dimensions, num_elements);
  }
  cudaDeviceSynchronize();
  gettimeofday(&tv2, NULL);
  
  cudaMemcpy(host_solution, device_solution, num_elements * num_elements * sizeof(float), cudaMemcpyDeviceToHost);
  cudaEventRecord(stopEvent,0);
  cudaEventElapsedTime(&ms, startEvent, stopEvent);
  cudaMemcpy(host_solution, device_solution, (num_elements*num_elements-num_elements)/2 * sizeof(float), cudaMemcpyDeviceToHost);
  printf("GPU1 solution: %12.0f \n", array_sum(host_solution, num_elements*num_elements));
  printf("Time needed by GPU1: %8.0f millisecs\n", double (tv2.tv_sec-tv1.tv_sec)*1000 + (tv2.tv_usec-tv1.tv_usec)/1000);
  print_cuda_error();
  printf ("%i blocks\n",blocks);
  printf ("%i grid_size.x\n",grid_size.x);
  
  
////////////////////////////////////////////////////////////////////////////////////////  
  gettimeofday(&tv1, NULL);
  for (int i = 0; i < repeats; i++) {
    distance2<<<blocks,grid_size.x>>>(device_solution, device_array, dimensions, num_elements);
  }
  cudaDeviceSynchronize();
  gettimeofday(&tv2, NULL);
  cudaMemcpy(host_solution, device_solution, num_elements * num_elements * sizeof(float), cudaMemcpyDeviceToHost);
  cudaEventRecord(stopEvent,0);
  cudaEventElapsedTime(&ms, startEvent, stopEvent);
  cudaMemcpy(host_solution, device_solution, (num_elements*num_elements-num_elements)/2 * sizeof(float), cudaMemcpyDeviceToHost);

  printf ("%i blocks\n",blocks);
  printf("%i grid_size.x\n", grid_size.x);
  printf("GPU2 solution: %12.0f \n", array_sum(host_solution, num_elements*num_elements));  
  printf("Time needed by GPU2 %8.0f millisecs\n", double (tv2.tv_sec-tv1.tv_sec)*1000 + (tv2.tv_usec-tv1.tv_usec)/1000);
  print_cuda_error();
  cudaMemcpy(device_solution, host_solution, (num_elements*num_elements-num_elements)/2 * sizeof(float), cudaMemcpyHostToDevice);
////////////////////////////////////////////////////////////////////////////////////////  
  makezero<<<grid_size.x, blocks>>>(device_solution, dimensions, num_elements);
  gettimeofday(&tv1, NULL);
  if ((num_elements*num_elements-num_elements)/2 < max_gridsize) {
    grid_size.x = num_elements*(num_elements-1)/2;
  } else {
    grid_size.x = max_gridsize;
  }
  blocks = ((num_elements*num_elements-num_elements)/2 + grid_size.x - 1)/grid_size.x;
  for (int i = 0; i < repeats; i++) {
    distance3<<<blocks,grid_size.x>>>(device_solution, device_array, dimensions, num_elements);
  }
  cudaDeviceSynchronize();
  gettimeofday(&tv2, NULL);
  
  cudaMemcpy(host_solution, device_solution, (num_elements*num_elements-num_elements)/2 * sizeof(float), cudaMemcpyDeviceToHost);
  printf ("\n%i blocks\n",blocks);
  printf ("%i grid_size.x\n",grid_size.x);
  printf("GPU3 solution: %12.0f \n", array_sum(host_solution, num_elements*(num_elements-1)/2));
  printf("Time needed by GPU3: %8.0f millisecs\n", double (tv2.tv_sec-tv1.tv_sec)*1000 + (tv2.tv_usec-tv1.tv_usec)/1000);
  print_cuda_error();
  //prints array
  printf("####array######\n");
  for(int i = 0; i < num_elements*dimensions && i < 60; i++)
  { 
    printf("%4.2f", host_array[i]);
    printf(".. ");
  }
  printf("\n");
  // prints solution
  printf("####solution###\n");
  for(int i = 0; i < 25 && i<(num_elements*num_elements-num_elements)/2; i++)
  {
    printf("%4.2f", host_solution[i]);
    printf(".. ");
  }
  for (int i = 0; i < num_elements; i++) {
    for (int j = 0; j < num_elements; j++) {
      host_solution[i * num_elements + j] = 0;
    }
  }
  gettimeofday(&tv1, NULL);
  //speed test CPU

  for (int r = 0; r < repeats/10; r++) {
    for (int i = 0; i < num_elements; i++) {
      for (int j = i + 1; j < num_elements; j++) {
        host_solution[i * num_elements + j] = 0;
        for (int d = 0; d < dimensions; d++) {
          host_solution[i * num_elements + j] = host_solution[i*num_elements + j] + (host_array[i*dimensions+d]-host_array[j*dimensions+d]) * (host_array[i*dimensions+d]-host_array[j*dimensions+d]);
        }
        host_solution[i * num_elements + j] = sqrt(host_solution[i * num_elements + j]);
      }
    }
  }
  for(int i = 0; i < 50 && i<(num_elements*num_elements-num_elements)/2; i++)
  {
    printf("%4.2f", host_solution[i]);
    printf(".. ");
  }
  gettimeofday(&tv2, NULL);
  printf("\n");
  printf("CPU solution: %12.0f \n", array_sum(host_solution, num_elements*num_elements));
  printf("Time needed by CPU: %8.0f millisecs\n", double (tv2.tv_sec-tv1.tv_sec)*10000 + (tv2.tv_usec-tv1.tv_usec)/100);
  
  //test whether GPU output is correct
  //for (int i = 0; i < num_elements; i++) {
    //for (int j = i; j < num_elements; j++) {
      //if (host_solution[i * num_elements + j] > 0.0000) {
        //printf("%f", host_solution[i*num_elements + j]);
        //printf("error\n");
      //}
    //}
  //}


  printf("centroid: \n");
  float * center_point;
  center_point = new float [dimensions - 1];
  center_point = centroid(host_array, num_elements * dimensions, dimensions);
  printf("%f \n", center_point[0]);
  printf("%f \n", center_point[1]);
  printf("%f \n", center_point[2]);
  
  
  
  // deallocate memory  
  free(host_array);
  free(host_solution);
  cudaFree(device_array);
  cudaFree(device_solution);
  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);
  
  printf("\ndone\n");
  return 0;
}






