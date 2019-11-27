
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
	int bx = blockIdx.x; // ����� ����� �� x
	int by = blockIdx.y; // ����� ����� �� y
	int tx = threadIdx.x; // ����� ���� � ����� �� x
	int ty = threadIdx.y; // ����� ���� � ����� �� y
	float sum = 0.0f;
	int ia = n * (BLOCK_SIZE * by + ty); // ����� ������ �� A�
	int ib = BLOCK_SIZE * bx + tx; // ����� ������� �� B�
	int ic = ia + ib; // ����� �������� �� ђ
	// ���������� �������� ������� C
	for (int k = 0; k < n; k++) sum += a[ia + k] * b[ib + k * n];
	c[ic] = sum;
}
int main()
{
	int N = 1024;
	int m, n, k;
	// �������� ����������-�������
	float timerValueGPU, timerValueCPU;
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	int numBytes = N * N * sizeof(float);
	
	float* adev, * bdev, * cdev, * a, * b, * bT, * c, * cc;
	// ��������� ������ �� host
	a = (float*)malloc(numBytes);
	b = (float*)malloc(numBytes); //������� B
	bT = (float*)malloc(numBytes); //����������������� ������� B
	c = (float*)malloc(numBytes); //������� � ��� GPU-��������
	cc = (float*)malloc(numBytes); //������� � ��� CPU-��������


	// ������� ������� A, B � ����������������� ������� B
	for (n = 0; n < N; n++)
	{
		for (m = 0; m < N; m++)
		{
			a[m + n * N] = 2.0f * m + n; b[m + n * N] = m - n; bT[m + n * N] = n - m;
		}
	}
	// ������� ����� ����� � ������
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(N / threads.x, N / threads.y);
	// ��������� ������ �� GPU
	cudaMalloc((void**)&adev, numBytes);
	cudaMalloc((void**)&bdev, numBytes);
	cudaMalloc((void**)&cdev, numBytes);
	// ---------------- GPU-������� ------------------------
// ����������� ������ A � B � host �� device
	cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);
	// ������ �������
	cudaEventRecord(start, 0);
	// ������ �������-����
	kernel_global <<<blocks, threads>>> (adev, bdev, N, cdev);
	// ������ ������� ���������� GPU-��������
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueGPU, start, stop);
	printf("\n GPU calculation time %f msec\n", timerValueGPU);
	// �����������, ����������� ������� C � device �� host
	cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);



	// -------------------- CPU-������� --------------------
	// ������ �������
	cudaEventRecord(start, 0);
	// ���������� ������� C
	for (n = 0; n < N; n++)
	{
		for (m = 0; m < N; m++)
		{
			cc[m + n * N] = 0.f;
			for (k = 0; k < N; k++) cc[m + n * N] += a[k + n * N] * bT[k + m * N]; // bT !!!
		}
	}
	// ������ ������� ���������� CPU-��������
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueCPU, start, stop);
	printf("\n CPU calculation time %f msec\n", timerValueCPU);
	//	printf("\n Rate %f x\n", timerValueCPU / timerValueGPU);

			// ������������ ������ �� GPU � CPU
	cudaFree(adev);
	cudaFree(bdev);
	cudaFree(cdev);
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(bT);
	cudaFreeHost(c);
	cudaFreeHost(cc);
	// ����������� ����������-�������
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}
