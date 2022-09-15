LIB_DIR = .
PATH:=/opt/conda/envs/rapids/bin/:${PATH}

default: metajsonparser

metajsonparser: setup.py *.pyx
	python3 -m pip install cython
	python3 setup.py build_ext --inplace && rm -f metajsonparser.c && rm -Rf build

clean:
	rm *.so

run_test:
	LD_LIBRARY_PATH=..:. PYTHONPATH=. python3 test/test.py test/sample_2000.json 2000
