"""
Full system evaluation: check lifted LTL formula and grounded propositions.
"""
import os
import argparse
import logging
from shutil import copy2
import spot

from ground import LOC2GID
from srer import PROPS, run_exp_srer
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
        ltl_true, ltl_out = true_out["lifted_ltl"], sys_out["lifted_ltl"]

        props_out = [prop for prop in PROPS if prop in ltl_out]
        for prop_out, prop in zip(props_out, PROPS):  # replace out of order props, e.g., G i h X G ! a -> G i b X G ! a
            ltl_out = ltl_out.replace(prop_out, prop)

        is_correct = True
        try:
            spot_correct = spot.are_equivalent(spot.formula(ltl_out), spot.formula(ltl_true))

            if not spot_correct and len(ltl_out) == len(ltl_true):  # invariant to order of propositions
                sre2prop_true = {sre.lower(): prop for prop, sre in zip(true_out["props"], true_out["sre_to_preds"].keys())}

                try:
                    prop_out2true = {f"<{prop}>": sre2prop_true[sre] for prop, sre in sys_out["lifted_symbol_map"].items()}

                    ltl_out_reorder = ltl_out
                    for prop in sys_out["lifted_symbol_map"].keys():
                        ltl_out_reorder = ltl_out_reorder.replace(prop, f"<{prop}>")
                    for prop_out, prop in prop_out2true.items():
                        ltl_out_reorder = ltl_out_reorder.replace(prop_out, prop)

                    spot_correct = spot.are_equivalent(spot.formula(ltl_out_reorder), spot.formula(ltl_true))

                except KeyError:  # SRER extracted incorrect SRE
                    spot_correct = False
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
            logging.info(f"Incorrect number of spatial referring expression:\ntrue: {true_ground_sps}\npred: {spg_ground_sps}")
            continue

        for sre_out, sps_topk_out in spg_ground_sps.items():
            if sre_out not in true_ground_sps:
                logging.info(f"Incorrect SRE:\ntrue: {list(true_ground_sps.keys())}\nnot contain pred: {sre_out}")
                is_correct = False
                break
            else:
                sp_true = true_ground_sps[sre_out][0]

                if not sps_topk_out:
                    logging.info(f"Incorrect spatila predicate grounding size empty:\n{sre_out}\n{spg_ground_sps}")
                    continue

                sp_out = sps_topk_out[0]

                if len(sp_true) != len(sp_out):
                    is_correct = False
                    logging.info(f"Incorrect spatial predicates size:\n{sre_out}\ntrue: {sp_true}\npred: {sp_out}")
                    break

                for (lmk_type_true, ground_true), (lmk_type_out, ground_out) in zip(sp_true.items(), sp_out.items()):
                    if lmk_type_out != lmk_type_true or ground_out != ground_true:
                        is_correct = False
                        break
                if not is_correct:
                    break

        if is_correct:
            ncorrects += 1
        else:
            logging.info("Incorrect full system output")

    logging.info(f"Full Accuracy: {ncorrects} / {len(true_outs)} = {ncorrects / len(true_outs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc", type=str, default="providence", choices=["providence", "boston", "auckland", "san_francisco"], help="env name.")
    parser.add_argument("--ablate", type=str, default="both", choices=["text", "image", "both", None], help="ablate out a modality.")
    parser.add_argument("--nsamples", type=int, default=3, help="number of sample utts per LTL formula or None for all")
    parser.add_argument("--seed", type=int, default=111, help="seed to random sampler.")  # 0, 1, 2, 42, 111 (resreved for ablate)
    parser.add_argument("--topk", type=int, default=10, help="top k most likely landmarks grounded by REG.")
    args = parser.parse_args()
    loc_id = f"{args.loc}_n{args.nsamples}_seed{args.seed}" if args.nsamples else f"{args.loc}_all_seed{args.seed}"

    data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
    graph_dpath = os.path.join(data_dpath, "maps", f"{LOC2GID[args.loc]}_ablate" if args.ablate else LOC2GID[args.loc])
    osm_fpath = os.path.join(data_dpath, "osm_ablate" if args.ablate else "osm", f"{args.loc}.json")
    utts_fpath = os.path.join(data_dpath, "dataset", f"{args.loc}_ablate" if args.ablate else f"{args.loc}", f"{loc_id}_utts.txt")
    model_fpath = os.path.join(os.path.expanduser("~"), "ground", "models", "checkpoint-best")
    rel_embeds_fpath = os.path.join(data_dpath, f"known_rel_embeds.json")
    reg_in_cache_fpath = os.path.join(data_dpath, f"reg_in_cache_{args.loc}.pkl")
    results_dpath = os.path.join(os.path.expanduser("~"), "ground", f"results_full_ablate_{args.ablate}" if args.ablate else "results_full", loc_id)
    os.makedirs(results_dpath, exist_ok=True)
    srer_out_fname = "srer_outs.json"
    srer_out_fpath = os.path.join(results_dpath, srer_out_fname)
    reg_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "reg"))
    spg_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "spg"))
    lt_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "lt"))
    true_results_fpath = os.path.join(data_dpath, "dataset", f"{args.loc}_ablate" if args.ablate else args.loc, f"{loc_id}_true_results.json")

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(results_dpath, "eval_results_full.log"), mode='w'),
                            logging.StreamHandler()
                        ]
    )
    logging.info(f"***** Full System Evaluation Ablate {args.ablate}: {loc_id}\n" if args.ablate else f"***** Full System Evaluation: {loc_id}\n")
    logging.info(f"{graph_dpath}\n{osm_fpath}\n{utts_fpath}\n{true_results_fpath}\n{results_dpath}\n")

    # Spatial Referring Expression Recognition (SRER)
    srer_out_fpath_modular = os.path.join(os.path.expanduser("~"), "ground", f"results_modular_ablate_{args.ablate}" if args.ablate else "results_modular", loc_id, srer_out_fname)
    srer_out_fpath_ablate_txt = os.path.join(os.path.expanduser("~"), "ground", "results_full_ablate_text", loc_id, srer_out_fname)
    srer_out_fpath_ablate_img = os.path.join(os.path.expanduser("~"), "ground", "results_full_ablate_image", loc_id, srer_out_fname)
    if not os.path.isfile(srer_out_fpath) and os.path.isfile(srer_out_fpath_modular):  # same SRER output for exp_full and  exp_modular
        copy2(srer_out_fpath_modular, srer_out_fpath)
    elif not os.path.isfile(srer_out_fpath) and args.ablate and os.path.isfile(srer_out_fpath_ablate_txt):  # same SRER output for ablate text and ablate image
        copy2(srer_out_fpath_ablate_txt, srer_out_fpath)
    elif not os.path.isfile(srer_out_fpath) and args.ablate and os.path.isfile(srer_out_fpath_ablate_img):
        copy2(srer_out_fpath_ablate_img, srer_out_fpath)
    else:
        run_exp_srer(utts_fpath, srer_out_fpath)
    eval_srer(true_results_fpath, srer_out_fpath)

    # Referring Expression Grounding (REG)
    run_exp_reg(srer_out_fpath, graph_dpath, osm_fpath, args.topk, args.ablate, reg_out_fpath, reg_in_cache_fpath)
    eval_reg(true_results_fpath, args.topk, reg_out_fpath)

    # Spatial Predicate Grounding (SPG)
    run_exp_spg(reg_out_fpath, graph_dpath, osm_fpath, args.topk, rel_embeds_fpath, spg_out_fpath)
    eval_spg(true_results_fpath, args.topk, spg_out_fpath)

    # Lifted Translation (LT)
    run_exp_lt(spg_out_fpath, model_fpath, lt_out_fpath)
    eval_lt(true_results_fpath, lt_out_fpath)

    # Full system evaluation
    eval_full_system(true_results_fpath, lt_out_fpath)
