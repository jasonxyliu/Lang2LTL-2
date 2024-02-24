"""
Full system evaluation: check lifted LTL formula and grounded propositions.
"""
import os
import argparse

from ground import LOC2GID
from srer import run_exp_srer
from reg import run_exp_reg
from spg import run_exp_spg
from lt import run_exp_lt


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

    # Spatial Referring Expression Recognition (SRER)
    run_exp_srer(utts_fpath, srer_out_fpath)

    # Referring Expression Grounding (REG)
    run_exp_reg(srer_out_fpath, graph_dpath, osm_fpath, args.topk, args.ablate, reg_out_fpath)

    # Spatial Predicate Grounding (SPG)
    run_exp_spg(reg_out_fpath, graph_dpath, osm_fpath, args.topk, rel_embeds_fpath, spg_out_fpath)

    # Lifted Translation (LT)
    run_exp_lt(spg_out_fpath, model_fpath, lt_out_fpath)
