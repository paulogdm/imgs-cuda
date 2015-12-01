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

	unsigned char buffer[3]={0,0,0};
	int col = blockIdx.y * blockDim.y + threadIdx.x;
	int row = blockIdx.x * blockDim.x + threadIdx.y;


	if(row < row_limit && col < col_limit){
		getAverage(in, row, col, buffer, pixel_size, col_limit, row_limit);
		int index = getIndex(row, col, col_limit, pixel_size);
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
	cudaEvent_t start, stop; //eventos para tempo

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
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	//out recebe dimensoes e aloca espaco
	out = in->partialClone();
	
	//comeca o tempo
	cudaEventRecord(start);

	//sincroniza HOST e DEVICE
	cudaDeviceSynchronize();
	
	rows = in->getRows();
	cols = in->getCols();
	
	dim3 threadsPerBlock(32, 32);

	dim3 numBlocks(cols/threadsPerBlock.x + cols%threadsPerBlock.x, 
					rows/threadsPerBlock.y + rows%threadsPerBlock.y); 

	//Kernel
	smoothImageCUDA<<< numBlocks, threadsPerBlock >>>(out->getData(), in->getData(), in->getPixelSize(), cols, rows);
	cudaDeviceSynchronize();
	
	//contabiliza o stop
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	if(WRITE_IMAGE_OUT)
		writeImage(argv[2], out);

	printf("Time: %.4f\n", milliseconds);

	delete in;
	delete out;
	
	return 0;
}
