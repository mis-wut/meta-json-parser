import timeit
import sys
import cudf
import test

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage {sys.argv[0]} fname")

    fname = sys.argv[1]
    time_count = 0

    with open(fname) as f:
       lines = len(f.readlines())

    # WARRNING: first run cost, can impact the measurments
    # First run of cudf takes extra time
    # additionally this should level the cache impact
    df = cudf.read_json(fname, lines=True)
    
    # TODO: fix column names in demo
    columns = list(df.columns)

    # Start time measure 
    starttime = timeit.default_timer()

    df = cudf.DataFrame._from_data(*test.wrapped_test(fname, lines))
    df.columns = columns


    # Fill Null
    # Done in parser

    # cut score into bins
    #df["score"] = pd.cut(df["score"], bins=[-1000, 0,1,2,3,4,5,6,7,8,9,10,100, 1000])

    # hash subreddit and author 
    #df["subreddit"] = df["subreddit"].hash_values()
    #df["author"] = df["author"].hash_values()

    # hash author_flair_css_class and author_flair_text
    #df["author_flair_text"] = df["author_flair_text"].hash_values()
    #df["author_flair_css_class"] = df["author_flair_css_class"].hash_values()

    # lower body text  
    # done in parser

    #Force computation 
    cdf = df.memory_usage()

    # End time measure 
    time_count += timeit.default_timer() - starttime

    print(f"Elapsed time {time_count}")
    print(f"Final check:")
    print(f"   sum of 'score' column {df['score'].sum()}")
    print(f"   df.shape {df.shape}")
    df.to_json("/tmp/bad.json", lines = True, orient="records")

