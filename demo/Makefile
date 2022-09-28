META_DIR = ../meta_cudf/opt1/


DEMO_FILES = $(wildcard *.json)
DEMO_DIRS = $(DEMO_FILES:.json=)
DEMO_SCHEMA = $(DEMO_FILES:.json=/schema.cuh)
.PHONY: clean check


all: $(DEMO_DIRS) check $(DEMO_SCHEMA)


check: 
	for i in $(DEMO_DIRS); do if [ ! -f $$i ]; then echo "IMPORTANT: customize and copy $$i/raw.schema.json to $$i/schema.json"; fi; done


%: %.json
	mkdir -p $@
	genson $< > $@/raw.schema.json

%.cuh: %.json
	python3 json2cuda.py $< > $@

docker: $(TARGET)
	cp meta_def.cuh $(META_DIR) 
	cp $</schema.cuh $(META_DIR)/meta_def_schema.cuh
	docker build -t cython ..
