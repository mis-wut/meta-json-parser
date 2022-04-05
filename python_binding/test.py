import cudf
import test
import timeit
import sys

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage {sys.argv[0]} fname repeats (cudf|meta)")

    fname = sys.argv[1]
    time_meta = 0
    time_cudf = 0
    N = int(sys.argv[2])

    with open(fname) as f:
       lines = len(f.readlines())

    for i in range(N):
       if "meta" in sys.argv[3]:
            starttime = timeit.default_timer()
            df = test.wrapped_test(fname, lines)
            time_meta = timeit.default_timer() - starttime

       if "cudf" in sys.argv[3]:
            starttime = timeit.default_timer()
            df1 = cudf.io.json.read_json(fname, lines =True, orient='columns')
            time_cudf = timeit.default_timer() - starttime

    if "meta" in sys.argv[3]:
       print(f"META runs {N} total time {time_meta}")

    if "cudf" in sys.argv[3]:
       print(f"CUDF runs {N} total time {time_cudf}")

