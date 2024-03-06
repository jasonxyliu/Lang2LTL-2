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
from evaluate import eval_srer, eval_reg, eval_spg, eval_lt
from utils import load_from_file


def eval_full_system(true_results_fpath, lt_out_fpath):
    true_outs = load_from_file(true_results_fpath)
    sys_outs = load_from_file(lt_out_fpath)
    ncorrects = 0

    assert len(sys_outs) == len(true_outs), f"ERROR different numbers of samples\ntrue: {len(true_outs)}\npred: {len(sys_outs)}"

    for true_out, sys_out in zip(true_outs, sys_outs):
        assert sys_out["utt"] == true_out["utt"], f"ERROR different utterances:\ntrue: {true_out['utt']}\npred: {sys_out['utt']}"
        logging.info(f"* Command: {sys_out['utt']}")

        # Lifted LTL formula
        is_correct = True
        ltl_true, ltl_out = true_out["lifted_ltl"], sys_out["lifted_ltl"]

        try:
            spot_correct = spot.are_equivalent(spot.formula(ltl_true), spot.formula(ltl_out))

            if not spot_correct:  # invariant to order of propositions
                ltl_str_true, ltl_str_out = str(ltl_true), str(ltl_out)
                prop2sre_true, prop2sre_out = {}, {}

                for prop, sre in zip(true_out["props"], true_out["sre_to_preds"].keys()):
                    ltl_str_true = ltl_str_true.replace(prop, f"<{prop}>")
                    prop2sre_true[f"<{prop}>"] = sre
                for prop, sre in prop2sre_true.items():
                    ltl_str_true = ltl_str_true.replace(prop, sre.lower())

                for prop, sre in sys_out["lifted_symbol_map"].items():
                    ltl_str_out = ltl_str_out.replace(prop, f"<{prop}>")
                    prop2sre_out[f"<{prop}>"] = sre
                for prop, sre in prop2sre_out.items():
                    ltl_str_out = ltl_str_out.replace(prop, sre.lower())

                spot_correct = ltl_str_out == ltl_str_true
        except SyntaxError:
            logging.info(f"Incorrect lifted translation Syntax Error\ntrue: {ltl_true}\npred: {ltl_out}")
            continue

        if not spot_correct:
            logging.info(f"Incorrect lifted translation:\ntrue: {spot.formula(ltl_true)}\npred: {spot.formula(ltl_out)}")
            continue

        # Spatial referring expression grounding
        true_ground_sps = true_out["grounded_sps"]
        spg_ground_sps = sys_out["grounded_sps"]
        if len(spg_ground_sps) != len(true_ground_sps):
            logging.info(f"ERROR incorrect number of spatial referring expression:\ntrue: {true_ground_sps}\npred: {spg_ground_sps}")
            continue

        for sre_out, sps_topk_out in spg_ground_sps.items():
            if sre_out not in true_ground_sps:
                logging.info(f"ERROR incorrect SRE:\ntrue: {list(true_ground_sps.keys())}\nnot contain pred: {sre_out}")
                is_correct = False
                break
            else:
                sp_true = true_ground_sps[sre_out][0]
                sp_out = sps_topk_out[0]

                if len(sp_true) != len(sp_out):
                    is_correct = False
                    logging.info(f"ERROR spatial predicates have different sizes:\n{sre_out}\ntrue: {sp_true}\npred: {sp_out}")
                    break

                for (lmk_type_true, ground_true), (lmk_type_out, ground_out) in zip(sp_true.items(), sp_out.items()):
                    if lmk_type_out != lmk_type_true or ground_out != ground_true:
                        is_correct = False
                        break
                if not is_correct:
                    break

        if is_correct:
            ncorrects += 1

    logging.info(f"Full Accuracy: {ncorrects} / {len(true_outs)} = {ncorrects / len(true_outs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc", type=str, default="boston", choices=["blackstone", "boston", "auckland", "san_francisco"], help="env name.")
    parser.add_argument("--ablate", type=str, default=None, choices=["text", "image", None], help="ablate out a modality or None to use both.")
    parser.add_argument("--nsamples", type=int, default=2, help="provide an integer to use synthetic dataset otherwise None.")
    parser.add_argument("--seed", type=int, default=0, help="seed to random sampler.")  # 0, 1, 2, 42, 111
    parser.add_argument("--topk", type=int, default=10, help="top k most likely landmarks grounded by REG.")
    args = parser.parse_args()
    loc_id = f"{args.loc}_n{args.nsamples}_seed{args.seed}" if args.nsamples else f"{args.loc}"

    data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
    graph_dpath = os.path.join(data_dpath, "maps", LOC2GID[args.loc])
    osm_fpath = os.path.join(data_dpath, "osm", f"{args.loc}.json")
    utts_fpath = os.path.join(data_dpath, "dataset", args.loc, f"{loc_id}_utts.txt")
    model_fpath = os.path.join(os.path.expanduser("~"), "ground", "models", "checkpoint-best")
    results_dpath = os.path.join(os.path.expanduser("~"), "ground", "results_full", loc_id)
    os.makedirs(results_dpath, exist_ok=True)
    rel_embeds_fpath = os.path.join(results_dpath, f"known_rel_embeds.json")
    srer_out_fname = f"srer_outs_ablate_{args.ablate}.json" if args.ablate else f"srer_outs.json"
    srer_out_fpath = os.path.join(results_dpath, srer_out_fname)
    reg_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "reg"))
    spg_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "spg"))
    lt_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "lt"))
    true_results_fpath = os.path.join(data_dpath, "dataset", args.loc, f"{loc_id}_true_results.json")

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(results_dpath, f"eval_results_full.log"), mode='w'),
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
    eval_srer(true_results_fpath, srer_out_fpath)
    eval_reg(true_results_fpath, args.topk, reg_out_fpath)
    eval_spg(true_results_fpath, args.topk, spg_out_fpath)
    eval_lt(true_results_fpath, lt_out_fpath)
    eval_full_system(true_results_fpath, lt_out_fpath)
