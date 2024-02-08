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
        lifted_utt = srer_out["lifted_utt"]
        query = lifted_utt.translate(str.maketrans('', '', ',.'))
        lifted_ltl = lt_module.type_constrained_decode([query])[0]
        spg_out["lifted_ltl"] = lifted_ltl
        print(f"{lifted_utt}\n{lifted_ltl}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, default="boston", choices=["indoor_env_0", "alley", "blackstone", "boston", "auckland"], help="domain name.")
    args = parser.parse_args()

    location = args.location
    ablation = "text"  # "text", "image", None
    topk = 5  # top k most likely landmarks grounded by REG

    data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
    graph_dpath = os.path.join(data_dpath, "maps", LOC2GID[location])
    osm_fpath = os.path.join(data_dpath, "osm", f"{location}.json")
    model_fpath = os.path.join(os.path.expanduser("~"), "ground", "models", "checkpoint-best")
    utt_fpath = os.path.join(data_dpath, f"utts_{location}.txt")
    results_dpath = os.path.join(os.path.expanduser("~"), "ground", "results")
    srer_out_fname = f"srer_outs_{location}_ablate_{ablation}.json" if ablation else f"srer_outs_{location}.json"
    reg_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "reg"))
    spg_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "spg"))

    # Spatial Referring Expression Recognition (SRER)
    srer_out_fpath = os.path.join(results_dpath, srer_out_fname)
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
        reg(graph_dpath, osm_fpath, srer_outs, topk, ablation)
        save_to_file(srer_outs, reg_out_fpath)

    # Spatial Predicate Grounding (SPG)
    reg_outs = load_from_file(reg_out_fpath)
    if not os.path.isfile(spg_out_fpath):
        init(graph_dpath, osm_fpath)
        for reg_out in reg_outs:
            reg_out['spg_results'] = spg(reg_out, topk)
        save_to_file(reg_outs, spg_out_fpath)

    # Lifted Translation (LT)
    # spg_outs = load_from_file(spg_out_fpath)
    # lt(spg_outs, model_fpath)
    # save_to_file(spg_outs, os.path.join(results_dpath, srer_out_fname.replace("srer", "lt")))

    gtr_fpath = os.path.join(data_dpath, f"groundtruth_{location}.json")
    evaluate_spg(spg_out_fpath, gtr_fpath, topk)
