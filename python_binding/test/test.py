import cudf
import metajsonparser as mp
import timeit
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage {sys.argv[0]} fname lines")

    fname = sys.argv[1]
    lines = int(sys.argv[2])
    df = mp.read_json(fname, lines)
    print(df)


