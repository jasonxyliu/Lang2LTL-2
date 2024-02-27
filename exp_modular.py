"""
Modular-wise Evaluation: correct input to each module, check modular output.
"""
import os
import argparse
import logging
from collections import defaultdict
import spot

from ground import LOC2GID
from srer import run_exp_srer
from reg import run_exp_reg
from spg import run_exp_spg
from lt import run_exp_lt
from utils import load_from_file


def eval_srer(true_results_fpath, utts_fpath, srer_out_fpath):
    logging.info("***** Evaluating SRER Module")
    run_exp_srer(utts_fpath, srer_out_fpath)

    true_outs = load_from_file(true_results_fpath)
    srer_outs = load_from_file(srer_out_fpath)
    ncorrects = 0

    assert len(srer_outs) == len(true_outs), f"ERROR different numbers of samples:\ntrue: {len(true_outs)}\npred: {len(srer_outs)}"

    for true_out, srer_out in zip(true_outs, srer_outs):
        assert srer_out["utt"].strip() == true_out["utt"].strip(), f"ERROR different utterances:\ntrue: {true_out['utt']}\npred: {srer_out['utt']}"
        logging.info(f"* Command: {srer_out['utt']}")
        is_correct = True

        for (sre_true, preds_true), (sre_out, preds_out) in zip(true_out["sre_to_preds"].items(), srer_out["sre_to_preds"].items()):
            if sre_out.strip() != sre_true.strip():
                is_correct = False
                logging.info(f"Incorrect SREs\ntrue: {sre_true}\npred: {sre_out}")

            for (rel_true, res_true), (rel_out, res_out) in zip(preds_true.items(), preds_out.items()):
                if rel_out.strip() != rel_true.strip():
                    is_correct = False
                    logging.info(f"Incorrect spatial relation\ntrue: {rel_true}\npred: {rel_out}")
                if res_out.strip() != res_true.strip():
                    is_correct = False
                    logging.info(f"Incorrect REs\ntrue: {res_true}\npred: {res_out}")

        if srer_out["lifted_utt"] != true_out["lifted_utt"]:
            is_correct = False
            logging.info(f"Incorrect lifted utterances\ntrue: {true_out['lifted_utt']}\npred: {srer_out['lifted_utt']}")

        if is_correct:
            ncorrects += 1

        logging.info(f"\n")
    logging.info(f"SRER Accuracy: {ncorrects}/{len(true_outs)} = {ncorrects / len(true_outs)}\n\n")


def eval_reg(true_results_fpath, graph_dpath, osm_fpath, topk, ablate, reg_out_fpath):
    """
    Compute the top K accuracy of Referring Expression Grounding module.
    """
    logging.info("***** Evaluating REG Module")
    run_exp_reg(true_results_fpath, graph_dpath, osm_fpath, topk, ablate, reg_out_fpath)

    true_outs = load_from_file(true_results_fpath)
    reg_outs = load_from_file(reg_out_fpath)
    topk2acc = defaultdict(int)
    total_res = 0

    assert len(reg_outs) == len(true_outs), f"ERROR different numbers of samples\ntrue: {len(true_outs)}\npred: {len(reg_outs)}"

    for true_out, reg_out in zip(true_outs, reg_outs):
        assert reg_out["utt"] == true_out["utt"], f"ERROR different utterances:\ntrue: {true_out['utt']}\npred: {reg_out['utt']}"
        logging.info(f"* Command: {true_out['utt']}")

        for (sre_true, pred_true), (sre_out, pred_out) in zip(true_out["grounded_sre_to_preds"].items(), reg_out["grounded_sre_to_preds"].items()):
            if sre_out != sre_true:
                logging.info(f"ERROR different spatial referring expression:\ntrue: {sre_true}\npred: {sre_out}")

            if len(pred_true) != len(pred_out):
                logging.info(f"ERROR different numbers of REs\ntrue: {len(pred_true)}\npred: {len(pred_out)}")
                continue

            res_true = [score_re[0][1] for score_re in list(pred_true.values())[0]]
            res_out = [[score_ground[1] for score_ground in grounded_res] for grounded_res in list(pred_out.values())[0]]
            total_res += len(res_true)

            for re_true, res_topk in zip(res_true, res_out):
                for end_idx in range(1, topk+1):
                    if re_true in res_topk[:end_idx]:
                        topk2acc[end_idx] += 1

    for idx in range(1, topk+1):
        logging.info(f"REG Top-{idx} Accuracy: {topk2acc[idx]} / {total_res} = {topk2acc[idx] / total_res}")
    logging.info("\n\n")


def eval_spg(true_results_fpath, graph_dpath, osm_fpath, topk, rel_embeds_fpath, spg_out_fpath):
    """
    Compute the top K accuracy of Spatial Predicate Grounding module.
    """
    logging.info("***** Evaluating SPG Module")
    run_exp_spg(true_results_fpath, graph_dpath, osm_fpath, topk, rel_embeds_fpath, spg_out_fpath)

    true_outs = load_from_file(true_results_fpath)
    spg_outs = load_from_file(spg_out_fpath)
    topk2acc = defaultdict(int)
    total_sps = 0

    assert len(spg_outs) == len(true_outs), f"ERROR different numbers of samples\ntrue: {len(true_outs)}\npred: {len(spg_outs)}"

    for true_out, spg_out in zip(true_outs, spg_outs):
        assert spg_out["utt"] == true_out["utt"], f"ERROR different utterances:\ntrue: {true_out['utt']}\npred: {spg_out['utt']}"
        logging.info(f"* Command: {true_out['utt']}")

        total_sps += len(true_out["grounded_sps"])

        for (sre_true, sp_true), (sre_out, sp_topk_out) in zip(true_out["grounded_sps"].items(), spg_out["grounded_sps"].items()):
            if sre_out != sre_true:
                logging.info(f"ERROR different spatial referring expression:\ntrue: {sre_true}\npred: {sre_out}")
                breakpoint()

            for end_idx in range(1, topk+1):
                for sp_out in sp_topk_out[:end_idx]:
                    if len(sp_true[0]) != len(sp_out):
                        logging.info(f"ERROR different number of spatial predicates:\ntrue: {sp_true[0]}\npred: {sp_out}")
                        continue

                    is_correct = True
                    for (lmk_type_true, ground_true), (lmk_type_out, ground_out) in zip(sp_true[0].items(), sp_out.items()):
                        if lmk_type_out != lmk_type_true or not (set(ground_out) & set(ground_true)):
                            is_correct = False
                            if end_idx == 1:
                                logging.info(f"Incorrect Top-1 spatial predicate grounding: \n{sre_true}\ntrue: {lmk_type_true}; {ground_true}\npred: {lmk_type_out}; {ground_out}")

                    if is_correct:
                        topk2acc[end_idx] += 1

    for idx in range(1, topk+1):
        logging.info(f"SPG Top-{idx} Accuracy: {topk2acc[idx]} / {total_sps} = {topk2acc[idx] / total_sps}")
    logging.info("\n\n")


def eval_lt(true_results_fpath, model_fpath, lt_out_fpath):
    logging.info("***** Evaluating LT")
    run_exp_lt(true_results_fpath, model_fpath, lt_out_fpath)

    true_outs = load_from_file(true_results_fpath)
    lt_outs = load_from_file(lt_out_fpath)
    ncorrects = 0

    assert len(lt_outs) == len(true_outs), f"ERROR different numbers of samples\ntrue: {len(true_outs)}\npred: {len(lt_outs)}"

    for true_out, lt_out in zip(true_outs, lt_outs):
        assert lt_out["utt"] == true_out["utt"], f"ERROR different utterances:\ntrue: {true_out['utt']}\npred: {lt_out['utt']}"
        logging.info(f"* Command: {lt_out['utt']}")

        is_correct = True
        ltl_true, ltl_out = true_out["lifted_ltl"], lt_out["lifted_ltl"]

        try:
            spot_correct = spot.are_equivalent(spot.formula(ltl_true), spot.formula(ltl_out))
        except SyntaxError:
            is_correct = False
            logging.info(f"Incorrect lifted translation Syntax Error\ntrue: {ltl_true}\npred: {ltl_out}")

        if not spot_correct:
            is_correct = False
            logging.info(f"Incorrect lifted translation:\ntrue: {spot.formula(ltl_true)}\npred: {spot.formula(ltl_out)}")

        if is_correct:
            ncorrects += 1

    logging.info(f"LT Accuracy: {ncorrects} / {len(true_outs)} = {ncorrects / len(true_outs)}\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, default="spg", choices=["srer", "reg", "spg", "lt", "all"], help="domain name.")
    parser.add_argument("--location", type=str, default="boston", choices=["blackstone", "boston", "auckland"], help="domain name.")
    parser.add_argument("--ablate", type=str, default=None, choices=["text", "image", None], help="ablate out a modality or None to use both")
    parser.add_argument("--nsamples", type=int, default=2, help="number of samples per LTL formula used to create dataset.")
    parser.add_argument("--topk", type=int, default=5, help="top k most likely landmarks grounded by REG")
    args = parser.parse_args()
    loc_id = f"{args.location}_n{args.nsamples}"

    data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
    graph_dpath = os.path.join(data_dpath, "maps", LOC2GID[args.location])
    osm_fpath = os.path.join(data_dpath, "osm", f"{args.location}.json")
    utts_fpath = os.path.join(data_dpath, f"{loc_id}_utts.txt")
    model_fpath = os.path.join(os.path.expanduser("~"), "ground", "models", "checkpoint-best")
    results_dpath = os.path.join(os.path.expanduser("~"), "ground", "results_modular", loc_id)
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
                            logging.FileHandler(os.path.join(results_dpath, f"eval_results_{args.module}.log"), mode='w'),
                            logging.StreamHandler()
                        ]
    )
    logging.info(f"***** Modular-wise Evaluation on Dataset: {loc_id}")

    if args.module == "srer" or args.module == "all":
        eval_srer(true_results_fpath, utts_fpath, srer_out_fpath)

    if args.module == "reg" or args.module == "all":
        eval_reg(true_results_fpath, graph_dpath, osm_fpath, args.topk, args.ablate, reg_out_fpath)

    if args.module == "spg" or args.module == "all":
        eval_spg(true_results_fpath, graph_dpath, osm_fpath, args.topk, rel_embeds_fpath, spg_out_fpath)

    if args.module == "lt" or args.module == "all":
        eval_lt(true_results_fpath, model_fpath, lt_out_fpath)
