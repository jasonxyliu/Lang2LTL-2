import os
import argparse
from tqdm import tqdm

from ground import LOC2GID
from srer import srer
from reg import reg
from spg import init, spg
from lt_s2s_sup_tcd import Seq2Seq
from utils import load_from_file, save_to_file
from evaluate import evaluate_spg


def lt(spg_outs, model_fpath):
    lt_module = Seq2Seq(model_fpath, "t5-base")
    for spg_out in spg_outs:
        lifted_utt = spg_out["lifted_utt"]
        query = lifted_utt.translate(str.maketrans('', '', ',.'))
        lifted_ltl = lt_module.type_constrained_decode([query])[0]
        spg_out["lifted_ltl"] = lifted_ltl
        print(f"{lifted_utt}\n{lifted_ltl}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, default="boston", choices=["lab", "alley", "blackstone", "boston", "auckland"], help="domain name.")
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

    # Spatial Referring Expression Recognition (SRER)
    if not os.path.isfile(srer_out_fpath):
        srer_outs = []
        utts = load_from_file(utt_fpath)
        for utt in tqdm(utts, desc='Performing spatial referring expression recognition (SRER)...'):
            _, srer_out = srer(utt)
            srer_outs.append(srer_out)
        save_to_file(srer_outs, srer_out_fpath)

    # Referring Expression Grounding (REG)
    if not os.path.isfile(reg_out_fpath):
        srer_outs = load_from_file(os.path.join(results_dpath, srer_out_fname))
        reg(graph_dpath, osm_fpath, srer_outs, args.topk, args.ablate)
        save_to_file(srer_outs, reg_out_fpath)

    # Spatial Predicate Grounding (SPG)
    if not os.path.isfile(spg_out_fpath):
        reg_outs = load_from_file(reg_out_fpath)
        init(graph_dpath, osm_fpath)
        for reg_out in reg_outs:
            reg_out['spg_results'] = spg(reg_out, args.topk)
        save_to_file(reg_outs, spg_out_fpath)

    true_results_fpath = os.path.join(data_dpath, f"true_results_{args.location}.json")
    evaluate_spg(spg_out_fpath, true_results_fpath, args.topk)

    # Lifted Translation (LT)
    spg_outs = load_from_file(spg_out_fpath)
    lt(spg_outs, model_fpath)
    save_to_file(spg_outs, os.path.join(results_dpath, srer_out_fname.replace("srer", "lt")))
