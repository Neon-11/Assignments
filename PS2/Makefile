#
# This is a Makefile. It provides instructions for compiling the code.
#
CC  = g++
HDR = $(shell find ./src/ -name "*.H")
SRC = $(wildcard ./src/*.cpp)
BIN = $(subst ./src, ./bin, $(SRC:.cpp=))
OPT = $(subst ./src, ./bin, $(SRC:.cpp=-opt))


default: debug

help:
	@echo
	@echo                       "--------------------- EM 525 Makefile Options------------------------"
	@echo                       "- make        # compile all executables in debug mode. This produces"
	@$(foreach file,$(BIN),echo "              #   $(file:.cpp=)";)
	@echo                       "- make help   # print this help message and exit"
	@echo                       "- make eigen  # download the eigen library and store in"
	@echo                       "              #   src/eigen3"
	@echo                       "- make debug  # same as make with no arguments"
	@echo                       "- make opt    # compile with optimizations. This produces"
	@$(foreach file,$(OPT),echo "              #   $(file:.cpp=)";)
	@echo                       "- make clean  # purge all build files"
	@echo

eigen: 
	git clone https://gitlab.com/libeigen/eigen.git
	mv eigen src/eigen3

debug: $(BIN)
	@echo "Done"

opt: $(OPT)
	@echo "Done" 

bin/%-opt: src/%.cpp $(HDR)
	mkdir -p bin
	$(CC) -std=c++17 $< -o $@ -I ./src -O3 -Wall -Wno-sign-compare -Wfatal-errors

bin/%: src/%.cpp $(HDR)
	mkdir -p bin
	$(CC) -std=c++17 $< -o $@ -I ./src -g3 -O0 -Wall -Wno-sign-compare -Wfatal-errors

clean:
	rm -rf bin
