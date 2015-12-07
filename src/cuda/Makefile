# Compilation variables. Compilation flags.
#for special includes copy this -I/path/for/somehting
CC=g++
CUDA_CC=nvcc
INCLUDE_DIR=../cuda_include/
FLAGS=-I$(INCLUDE_DIR)
NVIDIA_FLAGS=-L/usr/local/cuda/lib64 -lcuda -lcudart -rdc=true -gencode arch=compute_30,code=sm_30 -arch compute_30 -code sm_30 -m 64 -Xptxas=-dlcm=cg

#File sources
SOURCES= $(wildcard $(INCLUDE_DIR)*.cu) $(wildcard *.cu)
OBJECTS=$(SOURCES:.cu=.o)
HEADERS=$(wildcard *.h)
TARGET_DIR=../../bin/
TARGET=$(TARGET_DIR)main_cuda

#Main
all: $(TARGET)
#	
$(TARGET): $(OBJECTS)
	$(CUDA_CC) -o $@ $^ $(FLAGS) $(NVIDIA_FLAGS)

%.o: %.cu %.h
	$(CUDA_CC) $(FLAGS) -c $< -o $@ $(NVIDIA_FLAGS)

%.o: %.cu
	$(CUDA_CC) $(FLAGS) -c $< -o $@ $(NVIDIA_FLAGS)

#clean .o and main
clean:
	rm -f *.o $(TARGET)
	rm -f $(INCLUDE_DIR)*.o

#Quick clean+all
rebuild: clean all
