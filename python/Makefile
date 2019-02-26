all: build

build: setup.py gaussian_process.cc ../include/gp.hh
	@rm -rfv build
	CC=g++ python setup.py build

install: build
	CC=g++ python setup.py install

clean:
	@rm -rfv build

