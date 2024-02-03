import os

from tqdm import tqdm
from utils import load_from_file, save_to_file

def evaluate_spg(spg_output_fpth, gtr_fpath, location, topk=3):
	# NOTE: this function will take in a JSON outputted by the SPG module and it will produce a report
	#		of how many matches it gets right.

	# -- we need to count the total number of SREs across all commands in the JSON file:
	total_sres = 0
	total_topk = {f'top-{x+1}': 0 for x in range(topk)}

	spg_output = load_from_file(spg_output_fpth)
	gtr_data = load_from_file(gtr_fpath)[location]

