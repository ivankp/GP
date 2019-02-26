DEP := setup.py gaussian_process.cc

all: build

build: $(DEP)
	rm -rf build
	CC=g++ python setup.py build

install: $(DEP)
	CC=g++ python setup.py install

