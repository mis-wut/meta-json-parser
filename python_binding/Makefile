LIB_DIR = .
PATH:=/opt/conda/envs/rapids/bin/:${PATH}

default: testcudf

testcudf: setup.py *.pyx
	python3 -m pip install cython
	python3 setup.py build_ext --inplace && rm -f testcudf.c && rm -Rf build

clean:
	rm *.so

test:
	LD_LIBRARY_PATH=..:. python3 test.py
