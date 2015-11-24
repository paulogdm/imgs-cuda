# Compilation variables. Compilation flags.
#for special includes copy this -I/path/for/somehting
CC=g++

INCLUDE_DIR=../include/
FLAGS=-lm -Wall -I$(INCLUDE_DIR) -fopenmp


#File sources
SOURCES=$(wildcard *.cpp) $(wildcard $(INCLUDE_DIR)*.cpp)
OBJECTS=$(SOURCES:.cpp=.o)
HEADERS=$(wildcard *.h)
TARGET_DIR=../../bin/
TARGET=$(TARGET_DIR)main_st

#Main
all: $(TARGET)
#	
$(TARGET): $(OBJECTS)
	$(CC) -o $@ $^ $(FLAGS)

#ADT (TAD em portugues)
%.o: %.cpp %.h
	$(CC) $(FLAGS) -c $< -o $@

#if no header
%.o: %.cpp
	$(CC) $(FLAGS) -c $<

#clean .o and main
clean:
	rm -f *.o $(TARGET)
	rm -f $(INCLUDE_DIR)*.o

#Quick clean+all
rebuild: clean all
