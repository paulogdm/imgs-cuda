#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <image.h>
#include <smooth.h>

#include <mpi.h>

#define EXEC_N_TIMES	10

#define WRITE_IMAGE_OUT		false

//openmpi defines
#define FIRST_TAG	10	//primeira bateria de troca de mensagens
#define SECOND_TAG	20	//segunda bateria de troca de mensagens


Image* readImage(const char *name){

	Image *buffer = createImage(name);

	if(buffer != NULL)
		buffer->readFile(name);

	return buffer;
}

void writeImage(const char *name, Image *out){
	if(out != NULL)
		out->writeFile(name);
}


void smoothImage(Image *out, Image *in, int in_start, int in_end){

	int pixelSize = in->getPixelSize();
	unsigned char* pixel_array = out->getData();
		
	#pragma omp parallel
	{
		unsigned char *buffer = (unsigned char*) calloc(pixelSize, sizeof(unsigned char));
		int index;

		#pragma omp for
		for (int i = in_start; i < in_end; i++){
				
			for (int j = 0; j < in->getCols(); j++){

				getAverage(in, i, j, buffer);
				index = getIndex(i-in_start, j, in->getCols(), pixelSize);
				
				for(int c=0; c < pixelSize; c++){
					pixel_array[c + index] = buffer[c];
				}			
			}
		}
	
		free(buffer); //liberando buffer
	}
}


/**
 * Envia para todos os nos o pedacos de dados
 * @param image      imagem para dados
 * @param n_process  numero de processos disponivel
 * @param chunk_size tamanho do pedaco
 */
void sendChunks(Image *image, int n_process, int chunk_size){

	unsigned char *send_pointer; //ponteiro para os dados
	int up_border = __STEP__; //borda de cima, precisamos enviar mais dados do q chunk_size
	int low_border = __STEP__; //borda de baixo, no ultimo precisa conter o resto dos pixels
	int y_from, y_to;
	int total_size;

	for (int i = 1; i < n_process; i++){ //loop dos processos
		
		if(i == n_process-1)
			low_border = image->getRows()%n_process;

		//start e end
		y_from = chunk_size*(i) - up_border;
		y_to = chunk_size*(i+1) + low_border;

		//tamanho total
		total_size = image->getRowsSize(y_to - y_from);

		//enviando o array
		send_pointer = image->getData(y_from);
		MPI_Send(send_pointer, total_size, MPI_UNSIGNED_CHAR, i, FIRST_TAG ,MPI_COMM_WORLD);
	}
}

/**
 * Recebe todos os pedacos das iamgens dos slaves
 * @param image      imagem
 * @param n_process  numero de process
 * @param chunk_size tamanho do trecho
 */
void receiveChunks(Image *image, int n_process, int chunk_size){
	
	unsigned char *data_pointer;
	int low_border = 0;
	int total_size;

	for (int i = 1; i < n_process; i++){
		
		if(i == n_process-1){
			//BUG AQUI, o +1 não deveria existir mas se não coloco dá pau
			low_border = (image->getRows())%n_process+1;
		}

		total_size = image->getRowsSize(chunk_size+low_border);

		data_pointer = image->getData(chunk_size*(i));

		MPI_Recv(data_pointer, total_size, MPI_UNSIGNED_CHAR, i, SECOND_TAG ,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
}

/**
 * Envia um chunk para o ROOT
 * @param image imagem
 */
void sendChunk(Image *image){

	int total_size = image->getRowsSize(image->getRows());
	printf("T%d\n", total_size);
	image->writeFile("test.ppm");
	MPI_Send(image->getData(), total_size, MPI_UNSIGNED_CHAR, 0, SECOND_TAG ,MPI_COMM_WORLD);
}

/**
 * Recebe um pedado do ROOT
 * @param image      imagem
 * @param n_process  numero de processos
 * @param my_rank    rank do processo
 * @param chunk_size tamanho do chunk
 * @param total_rows total de linhas
 * @param type       tipo da imagem
 */
void receiveChunk(Image *image, int n_process, int my_rank, int chunk_size, int total_rows){

	int total_size;
	int borders = __STEP__*2;

	if(my_rank == n_process - 1){
		borders = (total_rows%n_process) + __STEP__;
	}

	total_size = image->getRowsSize(chunk_size+borders);


	MPI_Recv(image->getData(), total_size, MPI_UNSIGNED_CHAR, 0, FIRST_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

/**
 * Simples funcao para processar imagem
 * @param  image     imagem a ser processada
 * @param  n_process numero de processos
 * @param  rank      rank do processo
 * @return           imagem nova
 */
void processChunk(Image *out, Image *in, int n_process, int rank){

	if(rank==0){
		smoothImage(out, in, 0, (in->getRows()/n_process));
	}else if(rank == n_process-1){
		smoothImage(out, in, __STEP__, in->getRows());
	} else {
		smoothImage(out, in, __STEP__, in->getRows() - __STEP__);
	}

	return;
}

int main(int argc, const char **argv){

	Image *in;
	Image *out = NULL;
	int type, my_rank, n_process, total_rows;
	int chunk_size, cols;

	if(argc != 3){
		printf("Usage: %s <IMAGE_IN> <IMAGE_OUT>\n", argv[0]);
		return 1;
	}

	//iniciando MPI
	MPI_Init(&argc, (char***) &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &n_process);
	
	//ROOT calcula parametros iniciais
	if(my_rank == 0){
		in = readImage(argv[1]);
		
		if(in->getData() == NULL){
			printf("File does not seem to exist\n");
			return 1;
		}

		out = in->partialClone();
		
		type = getImageType(argv[1]);
		total_rows = in->getRows();
		chunk_size = total_rows / n_process;
		cols = in->getCols();
	}

	MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&total_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if(my_rank == 0){
		sendChunks(in, n_process, chunk_size);
	} else {
		if(my_rank == n_process - 1)
			in = createImage(type, chunk_size+__STEP__+(total_rows%n_process), cols);
		else
			in = createImage(type, chunk_size+2*__STEP__, cols);

		receiveChunk(in, n_process, my_rank, chunk_size, total_rows);

		if(my_rank == n_process - 1)
			out = createImage(type, chunk_size + (total_rows%n_process), cols);
		else 
			out = createImage(type, chunk_size, cols);
	}

	processChunk(out, in, n_process, my_rank);
	
	if(my_rank != 0){
		sendChunk(out);
	} else {
		receiveChunks(out, n_process, chunk_size);
		if(WRITE_IMAGE_OUT)
			writeImage(argv[2], out);
	}


	delete in;
	delete out;
	
	MPI_Finalize();

	return 0;
}
