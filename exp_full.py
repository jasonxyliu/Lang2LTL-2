"""
Full system evaluation: check lifted LTL formula and grounded propositions.
"""
import os
import argparse
import logging
import spot

from ground import LOC2GID
from srer import run_exp_srer
from reg import run_exp_reg
from spg import run_exp_spg
from lt import run_exp_lt
from utils import load_from_file


def eval_full_system(true_results_fpath, lt_out_fpath):
    true_outs = load_from_file(true_results_fpath)
    sys_outs = load_from_file(lt_out_fpath)
    ncorrects = 0

    assert len(sys_outs) == len(true_outs), f"ERROR different numbers of samples\ntrue: {len(true_outs)}\npred: {len(sys_outs)}"

    for true_out, sys_out in zip(true_outs, sys_outs):
        assert sys_out["utt"] == true_out["utt"], f"ERROR different utterances:\ntrue: {true_out['utt']}\npred: {sys_out['utt']}"
        logging.info(f"* Command: {sys_out['utt']}")

        is_correct = True

        # Lifted LTL formula
        ltl_true, ltl_out = true_out["lifted_ltl"], sys_out["lifted_ltl"]
        try:
            spot_correct = spot.are_equivalent(spot.formula(ltl_true), spot.formula(ltl_out))
        except SyntaxError:
            is_correct = False
            logging.info(f"Incorrect lifted translation Syntax Error\ntrue: {ltl_true}\npred: {ltl_out}")

        if not spot_correct:
            is_correct = False
            logging.info(f"Incorrect lifted translation:\ntrue: {spot.formula(ltl_true)}\npred: {spot.formula(ltl_out)}")

        if not is_correct:
            continue

        # Spatial referring expression grounding
        for (sre_true, sp_true), (sre_out, sp_topk_out) in zip(true_out["grounded_sps"].items(), sys_out["grounded_sps"].items()):
            if sre_out != sre_true:
                is_correct = False
                logging.info(f"ERROR different spatial referring expression:\ntrue: {sre_true}\npred: {sre_out}")

            sp_out = sp_topk_out[0]
            if len(sp_true[0]) != len(sp_out):
                is_correct = False
                logging.info(f"ERROR different number of spatial predicates:\ntrue: {sp_true[0]}\npred: {sp_out}")

            for (lmk_type_true, ground_true), (lmk_type_out, ground_out) in zip(sp_true[0].items(), sp_out.items()):
                if lmk_type_out != lmk_type_true or ground_out != ground_true:
                    is_correct = False
                    break

            if not is_correct:
                break

        if is_correct:
            ncorrects += 1

    logging.info(f"Accuracy: {ncorrects} / {len(true_outs)} = {ncorrects / len(true_outs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, default="boston", choices=["lab", "alley", "blackstone", "boston", "auckland"], help="env name.")
    parser.add_argument("--ablate", type=str, default=None, choices=["text", "image", None], help="ablate out a modality or None to use both.")
    parser.add_argument("--nsamples", type=int, default=None, help="provide an integer to use synthetic dataset otherwise None.")
    parser.add_argument("--topk", type=int, default=5, help="top k most likely landmarks grounded by REG.")
    args = parser.parse_args()
    loc_id = f"{args.location}_n{args.nsamples}" if args.nsamples else f"{args.location}"

    data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
    graph_dpath = os.path.join(data_dpath, "maps", LOC2GID[args.location])
    osm_fpath = os.path.join(data_dpath, "osm", f"{args.location}.json")
    utts_fpath = os.path.join(data_dpath, f"{loc_id}_utts.txt")
    model_fpath = os.path.join(os.path.expanduser("~"), "ground", "models", "checkpoint-best")
    results_dpath = os.path.join(os.path.expanduser("~"), "ground", "results_full", loc_id)
    os.makedirs(results_dpath, exist_ok=True)
    rel_embeds_fpath = os.path.join(results_dpath, f"known_rel_embeds.json")
    srer_out_fname = f"srer_outs_ablate_{args.ablate}.json" if args.ablate else f"srer_outs.json"
    srer_out_fpath = os.path.join(results_dpath, srer_out_fname)
    reg_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "reg"))
    spg_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "spg"))
    lt_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "lt"))
    true_results_fpath = os.path.join(data_dpath, f"{loc_id}_true_results.json")

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(results_dpath, f"eval_results.log"), mode='w'),
                            logging.StreamHandler()
                        ]
    )
    logging.info(f"***** Full System Evaluation Dataset: {loc_id}")

    # Spatial Referring Expression Recognition (SRER)
    run_exp_srer(utts_fpath, srer_out_fpath)

    # Referring Expression Grounding (REG)
    run_exp_reg(srer_out_fpath, graph_dpath, osm_fpath, args.topk, args.ablate, reg_out_fpath)

    # Spatial Predicate Grounding (SPG)
    run_exp_spg(reg_out_fpath, graph_dpath, osm_fpath, args.topk, rel_embeds_fpath, spg_out_fpath)

    # Lifted Translation (LT)
    run_exp_lt(spg_out_fpath, model_fpath, lt_out_fpath)

    # Full system evaluation
    eval_full_system(true_results_fpath, lt_out_fpath)
