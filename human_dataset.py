"""
Generate dataset of natural language commands collected from humans using Google Form.
"""
import os
import argparse
from pathlib import Path

from utils import load_from_file, save_to_file


def create_human_dataset(csv_fpath):
    csv_df = load_from_file(fpath=csv_fpath, use_pandas=True)
    location = Path(csv_fpath).stem.split("_")[-1].lower()
    human_utts_fpath = os.path.join(os.path.dirname(csv_fpath), f"{location}_human_utts.txt")
    utts = []

    for col_name, col_data in csv_df.items():
        if "Command" in col_name:  # column with "Command" in title refers to utterance for dataset
            utts.extend(col_data)
    save_to_file(data="\n".join(utts), fpth=human_utts_fpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_fpath", type=str, default=None, help="path to Google Form's CSV file.")
    args = parser.parse_args()

    create_human_dataset(args.csv_fpath)
