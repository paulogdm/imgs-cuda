#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <time.h>

#include <image.h>
#include <smooth.h>

#include <cuda.h>
#include <cuda_runtime.h>

//flag para executar o programa N vezes
#define EXEC_N_TIMES	3

//flag para escrever a saida ou nao
#define WRITE_IMAGE_OUT		/*!*/false

/**
 * Funcao que apenas auxiliar a leitura da imagem
 * @param  name nome do arquivo
 * @return      Uma classe imagem ou NULL se nao existir arquivo
 */
Image* readImage(const char *name){

	Image *buffer = NULL;
	
	//cria a imagem a partir do nome
	buffer = createImage(name);
	
	//le imagem se for valida
	if(buffer != NULL)
		buffer->readFile(name);

	//retorna imagem ou NULL se arquivo nao existir
	return buffer;
}

/**
 * Escreve o arquivo
 * @param name nome do arquivo
 * @param out  classe imagem
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

	//gridDim.x = blockDim.x * nBlocks.x
	//gridDim.y = blockDim.y * nBlocks.y
	int col = blockIdx.y * blockDim.y + threadIdx.x;
	int row = blockIdx.x * blockDim.x + threadIdx.y;

	//se a thread passou do limite. Especialmente util para borda inferior e da direita
	if(row < row_limit && col < col_limit){
		// buffer para receber as medias de cada cor
		unsigned char buffer[3]={0,0,0};

		//recebendo as medias
		getAverage(in, row, col, buffer, pixel_size, col_limit, row_limit);

		//indice 1D do ponteiro da imagem (aritmetica de ponteiros)
		int index = getIndex(row, col, col_limit, pixel_size);

		//Eficiencia melhorada em 5% com o pragma
		//for para o tamanho do pixel (1 se grayscale, 3 se RGB)
		#pragma unroll
		for(int c = 0; c < pixel_size; c++){
			out[c + index] = buffer[c];
		}
	}
}

/**
 * Funcao auxiliar para dar launch no kernel
 * @param in   input
 * @param out  imagem de output
 * @param name nome da imagem (apenas para printf)
 */
void lauchKernel(Image *in, Image* out, const char *name){

	cudaEvent_t evnt1, evnt2, evnt3, evnt4; //eventos para tempo

	//por legibilidade temos dimensoes
	int rows = in->getRows();
	int cols = in->getCols();

	//inicializando tempos
	cudaEventCreate(&evnt1);
	cudaEventCreate(&evnt2);
	cudaEventCreate(&evnt3);
	cudaEventCreate(&evnt4);
	
	//Calculando tamanho do grid (Blocks) e tamanho dos Blocos (Threads)
	//O grid eh o conjunto dos blocos enquanto o conjunto das threads eh o bloco
	dim3 Threads(32, 32);
	dim3 Blocks(ceil((float)rows/Threads.x),
					ceil((float)cols/Threads.y));
	
	//executando N vezes
	for(int	n_exec = 0; n_exec < EXEC_N_TIMES; n_exec++){

		cudaEventRecord(evnt1);
		//sync na memoria
		cudaDeviceSynchronize();

		cudaEventRecord(evnt2);
		//Kernel launch
		smoothImageCUDA<<< Blocks, Threads>>>(out->getData(), in->getData(), 
											in->getPixelSize(), cols, rows);

		cudaEventRecord(evnt3);
		//sync na memoria
		cudaDeviceSynchronize();
		cudaEventRecord(evnt4);
		
		//sync nos eventos
		cudaEventSynchronize(evnt3);
		cudaEventSynchronize(evnt4);
		
		//computando tempos
		float kernel_time = 0;
		float memory_time_in = 0;
		float memory_time_out = 0;

		cudaEventElapsedTime(&kernel_time, evnt2, evnt3);
		cudaEventElapsedTime(&memory_time_in, evnt1, evnt2);
		cudaEventElapsedTime(&memory_time_out, evnt3, evnt4);

		printf("Image: %s\n", name);
		printf("Kernel: %.4fms\n", kernel_time);
		printf("Memory in: %.4fms\n", memory_time_in);
		printf("Memory out: %.4fms\n", memory_time_out);
		printf("Total (K+M): %.4fms\n", kernel_time+memory_time_out+memory_time_in);
		printf("\n");

	}
}

/**
 * MAIN
 * @param  2: <nome da entrada> <nome da saida>
 */
int main(int argc, const char **argv){

	Image *in; //input
	Image *out; //output

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
	
	//out recebe dimensoes, tipo e aloca espaco
	out = in->partialClone();

	lauchKernel(in, out, argv[1]);
	
	//MUDAR A FLAG PARA ESCREVER ou nao
	if(WRITE_IMAGE_OUT)
		writeImage(argv[2], out);

	delete in;
	delete out;
	
	return 0;
}
