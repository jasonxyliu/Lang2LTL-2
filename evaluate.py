import os

from tqdm import tqdm
from utils import load_from_file, save_to_file

def evaluate_spg(spg_output_fpth, gtr_fpath, topk=3):
	# NOTE: this function will take in a JSON outputted by the SPG module and it will produce a report
	#		of how many matches it gets right.

	# -- we need to count the total number of SREs across all commands in the JSON file:
	total_sres = 0
	total_topk = {f'top-{x+1}': 0 for x in range(topk)}

	spg_output = load_from_file(spg_output_fpth)
	gtr_data = load_from_file(gtr_fpath)
	gtr_utt_to_results = {x['utt']: x for x in gtr_data}

	for result in spg_output:
		gtr = gtr_utt_to_results[result['utt']]
		total_sres += len(result['spg_results'])

		print(result['utt'])
		for P in result['spg_results']:
			gtr_spg = gtr['true_reg_spg'][result['spg_results'].index(P)]
			gtr_rel = list(gtr_spg.keys()).pop()
			gtr_target = gtr_spg[gtr_rel][0] if isinstance(gtr_spg[gtr_rel], list) else gtr_spg[gtr_rel]

			print(P)

			for k in range(min(topk, len(P['groundings']))):
				print(P['groundings'][k]['target'], gtr_target)
				if P['groundings'][k]['target'] == gtr_target:
					for j in range(k, topk):
						total_topk[f'top-{j+1}'] += 1
				break

	print(total_sres)
	print(total_topk)

	input()
