import os
import argparse
import logging
from collections import defaultdict
import spot

from utils import load_from_file


def eval_srer(true_results_fpath, srer_out_fpath):
	logging.info("***** Evaluating SRER")

	true_outs = load_from_file(true_results_fpath)
	srer_outs = load_from_file(srer_out_fpath)
	nincorrects = 0

	assert len(srer_outs) == len(true_outs), f"ERROR different numbers of samples: {len(true_outs)}, {len(srer_outs)}"

	for true_out, srer_out in zip(true_outs, srer_outs):
		assert srer_out["utt"] == true_out["utt"], f"ERROR different utterances:\n{true_out['utt']}\n{srer_out['utt']}"
		logging.info(f"* Command: {srer_out['utt']}")
		is_incorrect = False

		for (sre_true, preds_true), (sre_out, preds_out) in zip(true_out["srer_outs"].items(), srer_out["sre_to_preds"].items()):
			if sre_out != sre_true:
				is_incorrect = True
				logging.info(f"Incorrect SREs\ntrue: {sre_true}\npred: {sre_out}")
				# breakpoint()

			for (rel_true, res_true), (rel_out, res_out) in zip(preds_true.items(), preds_out.items()):
				if rel_out != rel_true:
					is_incorrect = True
					logging.info(f"Incorrect spatial relation\ntrue: {rel_true}\npred: {rel_out}")
					# breakpoint()
				if res_out != res_true:
					is_incorrect = True
					logging.info(f"Incorrect REs\ntrue: {res_true}\npred: {res_out}")
					# breakpoint()

		if srer_out["lifted_utt"] != true_out["lifted_utt"]:
			is_incorrect = True
			logging.info(f"Incorrect lifted utterances\ntrue: {true_out['lifted_utt']}\npred: {srer_out['lifted_utt']}")
			# breakpoint()

		if is_incorrect:
			nincorrects += 1

	logging.info(f"SRER Accuracy: {len(true_outs) - nincorrects}/{len(true_outs)} = {(len(true_outs) - nincorrects) / len(true_outs)}\n\n")


def eval_reg(true_results_fpath, reg_out_fpath, topk):
	"""
	Compute the top K accuracy of Referring Expression Grounding module.
	"""
	logging.info("***** Evaluating REG")

	true_outs = load_from_file(true_results_fpath)
	reg_outs = load_from_file(reg_out_fpath)
	topk2acc = defaultdict(int)
	total_res = 0

	assert len(reg_outs) == len(true_outs), f"ERROR different numbers of samples: {len(true_outs)}, {len(reg_outs)}"

	for true_out, reg_out in zip(true_outs, reg_outs):
		assert reg_out["utt"] == true_out["utt"], f"ERROR different utterances:\ntrue: {true_out['utt']}\npred: {reg_out['utt']}"
		logging.info(f"* Command: {true_out['utt']}")

		for (sre_true, preds_true), (sre_out, preds_out) in zip(true_out["reg_spg_outs"].items(), reg_out["grounded_sre_to_preds"].items()):
			if sre_out != sre_true:
				print(f"ERROR different spatial referring expression:\ntrue: {sre_true}\npred: {sre_out}")

			preds_out = list(preds_out.values())
			total_res += len(preds_out)

			for pred_true, preds_topk in zip(preds_true, preds_out):
				for end_idx in range(1, topk+1):
					for pred in pred_true:
						if pred in preds_topk[:end_idx]:
							topk2acc[end_idx] += 1

	for idx in range(1, topk+1):
		logging.info(f"REG Top-{idx} Accuracy: {topk2acc[idx]} / {total_res} = {topk2acc[idx] /total_res}")
	logging.info("\n\n")



def eval_spg(true_results_fpath, spg_out_fpth, topk):
	"""
	Compute the top K accuracy of Spatial Predicate Grounding module.
	"""
	total_sres = 0  # count the total number of SREs across all commands in the JSON file
	total_topk = {f'top-{x+1}': 0 for x in range(topk)}
	total_utts_correct = 0

	spg_outs = load_from_file(spg_out_fpth)
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


def eval_lt(true_results_fpath, lt_out_fpath):
	logging.info("***** Evaluating LT")

	true_outs = load_from_file(true_results_fpath)
	lt_outs = load_from_file(lt_out_fpath)
	utt2acc = {}
	nincorrects = 0

	assert len(lt_outs) == len(true_outs), f"ERROR different numbers of samples: {len(true_outs)}, {len(lt_outs)}"

	for true_out, lt_out in zip(true_outs, lt_outs):
		assert lt_out["utt"] == true_out["utt"], f"ERROR different utterances:\ntrue: {true_out['utt']}\npred: {lt_out['utt']}"
		logging.info(f"* Command: {lt_out['utt']}")
		is_incorrect = False

		ltl_true, ltl_out = true_out["lt_out"], lt_out["lifted_ltl"]

		try:
			spot_correct = spot.are_equivalent(spot.formula(ltl_true), spot.formula(ltl_out))
		except SyntaxError:
			logging.info(f"Incorrect lifted translation Syntax Error\ntrue: {ltl_true}\npred: {ltl_out}")
			is_incorrect = True

		if not spot_correct:
			is_incorrect = True
			logging.info(f"Incorrect lifted translation:\ntrue: {spot.formula(ltl_true)}\npred: {spot.formula(ltl_out)}")

		if is_incorrect:
			nincorrects += 1
			utt2acc[lt_out["utt"]] = 0
		else:
			utt2acc[lt_out["utt"]] = 1

	logging.info(f"LT Accuracy: {len(true_outs) - nincorrects} / {len(true_outs)} = {(len(true_outs) - nincorrects) / len(true_outs)}\n\n")
	return utt2acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, default="boston", choices=["blackstone", "boston", "auckland"], help="domain name.")
    parser.add_argument("--ablate", type=str, default=None, choices=["text", "image", None], help="ablate out a modality or None to use both")
    parser.add_argument("--nsamples", type=int, default=2, help="numbe of samples per LTL formula used to create dataset.")
    parser.add_argument("--topk", type=int, default=5, help="top k most likely landmarks grounded by REG")
    args = parser.parse_args()
    loc_id = f"{args.location}_n{args.nsamples}"

    data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
    utt_fpath = os.path.join(data_dpath, f"{loc_id}_utts.txt")
    results_dpath = os.path.join(os.path.expanduser("~"), "ground", "results")
    srer_out_fname = f"{loc_id}_srer_outs_ablate_{args.ablate}.json" if args.ablate else f"{loc_id}_srer_outs.json"
    srer_out_fpath = os.path.join(results_dpath, srer_out_fname)
    reg_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "reg"))
    spg_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "spg"))
    lt_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "lt"))
    true_results_fpath = os.path.join(data_dpath, f"{loc_id}_true_results.json")

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(results_dpath, f"{args.location}_eval_results.log"), mode='w'),
                            logging.StreamHandler()
                        ]
    )
    logging.info(f"***** Evaluating Dataset: {args.location}")

    eval_srer(true_results_fpath, srer_out_fpath)

    eval_reg(true_results_fpath, reg_out_fpath, args.topk)

    eval_spg(true_results_fpath, spg_out_fpath, args.topk)

    utt2acc = eval_lt(true_results_fpath, lt_out_fpath)
