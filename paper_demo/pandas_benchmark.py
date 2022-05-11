import timeit
import sys
import pandas as pd

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage {sys.argv[0]} fname")
        exit(0)

    fname = sys.argv[1]
    time_count = 0

    # Start time measure 
    starttime = timeit.default_timer()

    df = pd.read_json(fname, lines=True)

    # cut score into bins
    # df["score"] = pd.cut(df["score"], bins=[-1000, 0,1,2,3,4,5,6,7,8,9,10,100, 1000])

    # hash subreddit and author 
    df["subreddit"] = pd.util.hash_pandas_object(df["subreddit"])
    df["author"] = pd.util.hash_pandas_object(df["author"])
    df["distinguished"] = pd.util.hash_pandas_object(df["distinguished"])

    # hash author_flair_css_class and author_flair_text
    df["author_flair_text"] = pd.util.hash_pandas_object(df["author_flair_text"])
    df["author_flair_css_class"] = pd.util.hash_pandas_object(df["author_flair_css_class"])

    # lower body text  
    df["body"] = df["body"].str.lower()

    # End time measure 
    time_count += timeit.default_timer() - starttime

    print(f"Elapsed time {time_count}")
    print(f"Final check:")
    print(f"   sum of 'score' column {df['score'].sum()}")
    print(f"   df.shape {df.shape}")
