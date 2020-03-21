import pandas as pd
import argparse
from os import path

def get_args():
    parser = argparse.ArgumentParser(description="Create random data subsets for further computations")
    parser.add_argument("--train", help="Percentage of the traning subset to include", type=float)
    parser.add_argument("--valid", help="Percentage of the validation subset to include", type=float)
    parser.add_argument("--test", help="Percentage of the test subset to include", type=float)
    parser.add_argument("--prefix", help="Prefix of the names of the generated CSV files",
                        type=str, default="sun_subset_")
    parser.add_argument("--directory", help="Directory to save the generated CSV files", type=str, default="data")
    args = parser.parse_args()
    return vars(args)


def main(prefix, directory, **percentages):
    for subset in ["train", "valid", "test"]:
        if percentages[subset] is not None:
            df = pd.read_csv(f"data/sun_{subset}.csv").set_index("id")
            df = df.sample(frac=percentages[subset], replace=False)
            df.to_csv(path.join(directory, f"{prefix}{subset}.csv"))


if __name__ == '__main__':
    args = get_args()
    main(**args)