# Demo and benchmarks for DSSA 2022

1. Install genson (`sudo python3 -m pip install genson`).

2. Place a JSON file in this directory (has to have .json extension).

See reddit.json sample used in the paper.

3. Run `make`
 
First `make` run will setup directories for supplied JSON files and generate JSON schema files.
For each JSON file a directory will be crated containing file raw.schem.json.
In our reddit example - directory `reddit` and file `reddit/raw.schema.json` will be created.

4. Customize the schema and copy raw.schema.json to schema.json in each directory.

In our reddit example you can use supplied `reddit.schema` and copy it to `reddit/schema.json`.
If not sure of the customizations you can copy raw.schema.json to schema.json, but remember that this will use default conversion wich may be suboptimal. 

5. Run `make`

Second `make` run will generate schema.cuh for each directory associated with JSON files.
If you recive error most probabbly you miss `schema.json` file from step 4.
In our reddit example `reddit/schema.json` will be converted to `reddit/schema.cuh`.

6. Build docker image

In order to test our solution you need to compile meta-json-parser with libcudf support.
The simplest way is to use docker image prepared by RAPIDS.ai (libcudf is part of RAPIDS.ai).

We prepared a Dockerfile that will allow you to build 

make docker TARGET=reddit
