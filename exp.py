import os
from tqdm import tqdm

from ground import LOC2GID
from srer import srer
from reg import reg
from spg import init, spg
from lt_s2s_sup_tcd import Seq2Seq
from utils import load_from_file, save_to_file


def ground(lifted_utt, model_fpath):
    query = lifted_utt.translate(str.maketrans('', '', ',.'))
    lt_module = Seq2Seq(model_fpath, "t5-base")
    lifted_ltl = lt_module.type_constrained_decode([query])[0]
    return lifted_ltl


if __name__ == "__main__":
    location = "indoor_env_0"
    ablation = "text"  # "text", "image", None

    data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
    graph_dpath = os.path.join(data_dpath, "maps", LOC2GID[location])
    osm_fpath = os.path.join(data_dpath, "osm", f"{location}.json")
    model_fpath = os.path.join(os.path.expanduser("~"), "ground", "models", "checkpoint-best")
    utt_fpath = os.path.join(data_dpath, f"utts_{location}.txt")
    results_dpath = os.path.join(os.path.expanduser("~"), "ground", "results")
    srer_out_fname = f"srer_outs_{location}_ablate_{ablation}.json" if ablation else f"srer_outs_{location}.json"
    reg_out_fname = srer_out_fname.replace("srer", "reg")
    spg_out_fname = srer_out_fname.replace("srer", "spg")
    topk = 5  # top k most likely landmarks grounded by REG


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
    srer_outs = load_from_file(os.path.join(results_dpath, srer_out_fname))
    reg_outs = reg(graph_dpath, osm_fpath, srer_outs, topk, ablation)
    save_to_file(reg_outs, os.path.join(results_dpath, srer_out_fname.replace("srer", "reg")))


    # Spatial Predicate Grounding (SPG)
    reg_outs = load_from_file(os.path.join(results_dpath, reg_out_fname))
    init(graph_dpath, osm_fpath)

    spg_outs = []
    for reg_out in reg_outs:
        # -- make a copy of the dictionary for a corresponding utterance:
        spg_out = reg_out
        # -- add a new field to the dictionary with the final output of SPG (if any):
        spg_out['spg_results'] = spg(reg_out, topk)
        # spg_out.pop('grounded_spatial_preds')
        spg_outs.append(spg_out)

    save_to_file(spg_outs, os.path.join(results_dpath, srer_out_fname.replace("srer", "spg")))


    # Lifted Translation (LT)
    srer_outs = load_from_file(os.path.join(results_dpath, srer_out_fname))
    lt_outs = []

    for srer_out in srer_outs:
        lifted_utt = srer_out["lifted_utt"]
        lifted_ltl = ground(lifted_utt, model_fpath)

        srer_out["lifted_ltl"] = lifted_ltl
        lt_outs.append(srer_out)

        print(f"{lifted_utt}\n{lifted_ltl}\n")
        # breakpoint()

    save_to_file(lt_outs, os.path.join(results_dpath, srer_out_fname.replace("srer", "lt")))

    # lifted_utt = "go to a at most five times"
    # lifted_ltl = ground(lifted_utt, model_fpath)
    # print(lifted_ltl)
