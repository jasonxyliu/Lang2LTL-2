import os
import argparse

from ground import LOC2GID
from utils import load_from_file


def evaluate_spg(spg_outs_fpth, true_results_fpath, topk):
	"""
	Compute the top K accuracy of Spatial Predicate Grounding module.
	"""
	total_sres = 0  # count the total number of SREs across all commands in the JSON file
	total_topk = {f'top-{x+1}': 0 for x in range(topk)}
	total_utts_correct = 0

	spg_outs = load_from_file(spg_outs_fpth)
	true_results = load_from_file(true_results_fpath)
	gtr_utt_to_results = {x['utt']: x for x in true_results}

	for spg_out in spg_outs:
		print(f"Command: {spg_out['utt']}")
		gtr = gtr_utt_to_results[spg_out['utt']]

		spgs_gtr = gtr['true_reg_spg']
		spgs_gen = spg_out['spg_results']
		total_sres += len(spgs_gtr)

		top1_matches = 0

		for sre in spgs_gtr:
			print(f" >> SRE:\t\t{sre}")
			print(f" >> TRUE GROUNDING:\t{spgs_gtr[sre]}")

			spg_gen = spgs_gen[sre] if sre in spgs_gen else None
			print(f" >> COMPUTED GROUNDING:\t{[spg_gen[x]['target'] for x in range(len(spg_gen))]}")

			topk_match = False
			for k in range(min(topk, len(spg_gen))):
				if spg_gen[k]['target'] in spgs_gtr[sre]:
					print(f"   -> grounding found:\tk={k+1}!")
					topk_match = True

					# -- if we found a match in the top m results, then we found it in all top m to k:
					for j in range(k, topk):
						total_topk[f'top-{j+1}'] += 1

					if k == 0:
						top1_matches += 1

					break

			if not topk_match:
				print(f"   -> NO MATCH FOUND:\t{spg_gen[k]['target']}")
			print()

			if top1_matches == len(spgs_gtr):
				total_utts_correct += 1

		print('\n')

	print("*** SUMMARY: ***")
	for k in range(topk):
		print(f" top-{k+1} accuracy: {total_topk[f'top-{k+1}'] / total_sres:.5f}")
	print(f"\n* Total commands completely correct: {total_utts_correct}/{len(spg_outs)} ({total_utts_correct/len(spg_outs):.5})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, default="boston", choices=["blackstone", "boston", "auckland"], help="domain name.")
    parser.add_argument("--ablate", type=str, default=None, choices=["text", "image", None], help="ablate out a modality or None to use both")
    parser.add_argument("--topk", type=int, default=5, help="top k most likely landmarks grounded by REG")
    args = parser.parse_args()

    data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
    graph_dpath = os.path.join(data_dpath, "maps", LOC2GID[args.location])
    osm_fpath = os.path.join(data_dpath, "osm", f"{args.location}.json")
    model_fpath = os.path.join(os.path.expanduser("~"), "ground", "models", "checkpoint-best")
    utt_fpath = os.path.join(data_dpath, f"utts_{args.location}.txt")
    results_dpath = os.path.join(os.path.expanduser("~"), "ground", "results")
    srer_out_fname = f"srer_outs_{args.location}_ablate_{args.ablate}.json" if args.ablate else f"srer_outs_{args.location}.json"
    srer_out_fpath = os.path.join(results_dpath, srer_out_fname)
    reg_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "reg"))
    spg_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "spg"))


    true_results_fpath = os.path.join(data_dpath, f"{args.location}_true_results.json")
    evaluate_spg(spg_out_fpath, true_results_fpath, args.topk)
