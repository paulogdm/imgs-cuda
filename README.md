# Smooth de Imagens com CUDA

## Sinopse
	
## Autores:

    Oscar Lima Neto		oscarneto@usp.br
    Paulo G. De Mitri 	paulo.mitri@usp.br


## Compilar e executar

    Cada código fonte está na sua respectiva pasta em `src/`
    Portanto para compilar é necessário entrar na pasta desejada e aplicar o comando make
    Exemplo:
	    $ cd src/st/
	    $ make rebuild
    
    Para executar o script `run.sh` deve ser utilizado.
    Os seguintes argumentos são permitidos, caso seja necessário especificar uma 
    imagem ou um programa em específco:
    1) -fname <NOME DA IMAGEM>
    2) -bin <NOME DO PROGRAMA>
    Exemplos:
        $ ./run.sh -bin main_cuda
            (executando todas as imagens com o programa em CUDA)
        
        $./run.sh -fname CCCP.ppm
            (executando todos os programas com essa imagem)
        
        $./run.sh -bin main_st -fname EVGA.ppm
        $./run.sh -fname EVGA.ppm -bin main_st
            (executando a imagem EVGA.ppm com o programa main_st)

## Entrada e saída

    O programa retira todas as imagens de um único diretório: `in/`
    Toda a saída é colocada em `out/`
    Os tempos são apresentados com a extensão `.time`

## F.A.Q.

1) Quero alterar o número de execuções do programa...
É necessário editar o código fonte (main.cpp ou main.cu) e alterar a definição de EXEC_N_TIMES.

2) As imagens não estão sendo gravadas. O que aconteceu?
OU
3) Não preciso gerar as imagens, o que faço?
Existe uma flag em cada um dos arquivos 'main_*' chamada WRITE_IMG_OUT.
Altere ela para 'false' e o programa não irá gravar imagens. Altere para '!false' e o programa irá gravar imagens

4) Gostaria de otimizar a compição do algorítmo em CUDA para outra placa de vídeo...
Basta mudar as flags necessárias no Makefile do programa CUDA.