# Smooth de Imagens com CUDA

## Sinopse
	
##Autores:

Oscar Lima Neto		oscarneto@usp.br
Paulo G. De Mitri 	paulo.mitri@usp.br


## Compilar e executar

Cada código fonte está na sua respectiva pasta em src/
Portanto para compilar é necessário entrar na pasta desejada e aplicar o comando make
Exemplo:
	$ cd src/st/
	$ make rebuild

## Entrada e saída

O programa retira todas as imagens de um único diretório: `in/`
Toda a saída é colocada em `out/`
Os tempos são apresentados com a extensão `.time`

##F.A.Q.

1) Quero alterar o número de execuções do programa.
É necessário editar o código fonte (main.cpp ou main.cu) e alterar a definição de EXEC_N_TIMES.

2) As imagens não estão sendo gravadas. O que aconteceu?
OU
3) Não preciso gerar as imagens, o que faço?
Existe uma flag em cada um dos arquivos 'main_*' chamada WRITE_IMG_OUT.
Altere ela para 'false' e o programa não irá gravar imagens. Altere para '!false' e o programa irá gravar imagens
