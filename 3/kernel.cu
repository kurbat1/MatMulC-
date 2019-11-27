
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include "math.h"
#include "time.h"
#include <iostream>
#include <fstream>
#include <iomanip>

#define BLOCK_SIZE 32
__global__ void kernel_global(float* a, float* b, int n, float* c)
{
	int bx = blockIdx.x; // номер блока по x
	int by = blockIdx.y; // номер блока по y
	int tx = threadIdx.x; // номер нити в блоке по x
	int ty = threadIdx.y; // номер нити в блоке по y
	float sum = 0.0f;
	int ia = n * (BLOCK_SIZE * by + ty); // номер строки из A’
	int ib = BLOCK_SIZE * bx + tx; // номер столбца из B’
	int ic = ia + ib; // номер элемента из С’
	// вычисление элемента матрицы C
	for (int k = 0; k < n; k++) sum += a[ia + k] * b[ib + k * n];
	c[ic] = sum;
}
int main()
{
	int N = 1024;
	int m, n, k;
	// создание переменных-событий
	float timerValueGPU, timerValueCPU;
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	int numBytes = N * N * sizeof(float);
	
	float* adev, * bdev, * cdev, * a, * b, * bT, * c, * cc;
	// выделение памяти на host
	a = (float*)malloc(numBytes);
	b = (float*)malloc(numBytes); //матрица B
	bT = (float*)malloc(numBytes); //транспонированная матрица B
	c = (float*)malloc(numBytes); //матрица С для GPU-варианта
	cc = (float*)malloc(numBytes); //матрица С для CPU-варианта


	// задание матрицы A, B и транспонированной матрицы B
	for (n = 0; n < N; n++)
	{
		for (m = 0; m < N; m++)
		{
			a[m + n * N] = 2.0f * m + n; b[m + n * N] = m - n; bT[m + n * N] = n - m;
		}
	}
	// задание сетки нитей и блоков
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(N / threads.x, N / threads.y);
	// выделение памяти на GPU
	cudaMalloc((void**)&adev, numBytes);
	cudaMalloc((void**)&bdev, numBytes);
	cudaMalloc((void**)&cdev, numBytes);
	// ---------------- GPU-вариант ------------------------
// копирование матриц A и B с host на device
	cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);
	// запуск таймера
	cudaEventRecord(start, 0);
	// запуск функции-ядра
	kernel_global <<<blocks, threads>>> (adev, bdev, N, cdev);
	// оценка времени вычисления GPU-варианта
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueGPU, start, stop);
	printf("\n GPU calculation time %f msec\n", timerValueGPU);
	// копирование, вычисленной матрицы C с device на host
	cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);



	// -------------------- CPU-вариант --------------------
	// запуск таймера
	cudaEventRecord(start, 0);
	// вычисление матрицы C
	for (n = 0; n < N; n++)
	{
		for (m = 0; m < N; m++)
		{
			cc[m + n * N] = 0.f;
			for (k = 0; k < N; k++) cc[m + n * N] += a[k + n * N] * bT[k + m * N]; // bT !!!
		}
	}
	// оценка времени вычисления CPU-варианта
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueCPU, start, stop);
	printf("\n CPU calculation time %f msec\n", timerValueCPU);
	//	printf("\n Rate %f x\n", timerValueCPU / timerValueGPU);

			// освобождение памяти на GPU и CPU
	cudaFree(adev);
	cudaFree(bdev);
	cudaFree(cdev);
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(bT);
	cudaFreeHost(c);
	cudaFreeHost(cc);
	// уничтожение переменных-событий
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}
