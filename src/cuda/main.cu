#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <time.h>

#include <image.h>
#include <smooth.h>

#include <cuda.h>
#include <cuda_runtime.h>

//flag para escrever a saida ou nao
#define WRITE_IMAGE_OUT		!false

/**
 * Funcao que apenas auxiliar a leitura da imagem
 * @param  name nome do arquivo
 * @return      Uma classe imagem
 */
Image* readImage(const char *name){

	Image *buffer = NULL;
	
	buffer = createImage(name);
	
	if(buffer != NULL)
		buffer->readFile(name);

	return buffer;
}

/**
 * Escreve o arquivo
 * @param name nome do arquivo
 * @param out  classe da imagem
 */
void writeImage(const char *name, Image *out){
	if(out != NULL)
		out->writeFile(name);
}

/**
 * Kernel para smooth em CUDA
 * @param out        ponteiro para os dados de saida
 * @param in         ponteiro para os dados de entrada
 * @param pixel_size tamanho do pixel
 * @param col_limit  quantidade de colunas
 * @param row_limit  quantidade de linhas
 */
__global__ void smoothImageCUDA(unsigned char *out, unsigned char *in, int pixel_size, int col_limit, int row_limit){

	int col = blockIdx.y * blockDim.y + threadIdx.x;
	int row = blockIdx.x * blockDim.x + threadIdx.y;

	if(row < row_limit && col < col_limit){
		unsigned char buffer[3]={0,0,0};
		getAverage(in, row, col, buffer, pixel_size, col_limit, row_limit);
		int index = getIndex(row, col, col_limit, pixel_size);

		#pragma unroll
		for(int c = 0; c < pixel_size; c++){
			out[c + index] = buffer[c];
		}
	}
}

/**
 * MAIN
 * @param  2: <arquivo de entrada> <nome da saida>
 */
int main(int argc, const char **argv){

	Image *in;
	Image *out;
	int rows; //linhas
	int cols; //colunas
	cudaEvent_t evnt1, evnt2, evnt3, evnt4; //eventos para tempo

	if(argc != 3){
		printf("Usage: %s <IMAGE_IN> <IMAGE_OUT>\n", argv[0]);
		return 1;
	}

	//lendo imagem
	in = readImage(argv[1]);

	//se arquivo nao existe
	if(in->getData() == NULL){
		printf("File does not seem to exist\n");
	}

	//inicializando tempos
	cudaEventCreate(&evnt1);
	cudaEventCreate(&evnt2);
	cudaEventCreate(&evnt3);
	cudaEventCreate(&evnt4);
	
	//out recebe dimensoes e aloca espaco
	out = in->partialClone();
	
	//comeca o tempo

	//sincroniza HOST e DEVICE
	
	rows = in->getRows();
	cols = in->getCols();
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(ceil((float)rows/threadsPerBlock.x),
					ceil((float)cols/threadsPerBlock.y));
	

	//Kernel
	
	cudaEventRecord(evnt1);
	cudaDeviceSynchronize();

	cudaEventRecord(evnt2);
	smoothImageCUDA<<< numBlocks, threadsPerBlock>>>(out->getData(), in->getData(), in->getPixelSize(), cols, rows);

	cudaEventRecord(evnt3);
	cudaDeviceSynchronize();
	cudaEventRecord(evnt4);
	
	cudaEventSynchronize(evnt3);
	cudaEventSynchronize(evnt4);
	
	float kernel_time = 0;
	float memory_time_in = 0;
	float memory_time_out = 0;

	cudaEventElapsedTime(&kernel_time, evnt2, evnt3);
	cudaEventElapsedTime(&memory_time_in, evnt1, evnt2);
	cudaEventElapsedTime(&memory_time_out, evnt3, evnt4);

	if(WRITE_IMAGE_OUT)
		writeImage(argv[2], out);

	printf("\n");
	printf("Image: %s\n", argv[1]);
	printf("Kernel: %.4fms\n", kernel_time);
	printf("Memory in: %.4fms\n", memory_time_in);
	printf("Memory out: %.4fms\n", memory_time_out);
	printf("Total (K+M): %.4fms\n", kernel_time+memory_time_out+memory_time_in);
	printf("\n");

	delete in;
	delete out;
	
	return 0;
}
