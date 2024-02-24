import os
import argparse

from utils import load_from_file, save_to_file

def create_human_dataset(csv_fpath,):
    # -- load CSV file given as file path:
    csv_df = load_from_file(fpath=csv_fpath, use_pandas=True)
    location = str(os.path.splitext(os.path.basename(csv_fpath))[0].split("_")[-1]).lower()
    human_utts_fpath = os.path.join(os.path.dirname(csv_fpath), f"{location}_human_utts.txt")

    utts_for_file = []

    for (columnName, columnData) in csv_df.items():
        # NOTE: each column with "Command" in its title refers to that containing utterances for file generation:
        if "Command" in columnName:
            utts_for_file.extend(columnData)

    save_to_file(data="\n".join(utts_for_file), fpth=human_utts_fpath)
#enddef


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="path to CSV file from Google Forms.")
    args = parser.parse_args()

    create_human_dataset(args.csv)