import os
from tqdm import tqdm

from srer import srer
from reg import reg
from spg import init, spg
from lt_s2s_sup_tcd import Seq2Seq
from utils import load_from_file, save_to_file


loc2gid = {
    "alley": "downloaded_graph_2024-02-02_14-26-54",
    "indoor_env_0": "downloaded_graph_2024-02-02_10-55-35",
    "blackstone": "downloaded_graph_2024-01-27_07-48-53",
    "boston": "downloaded_graph_2024-01-27_07-48-53",
    "auckland": "",
}  # location to Spot graph ID


def ground(lifted_utt, model_fpath):
    query = lifted_utt.translate(str.maketrans('', '', ',.'))
    lt_module = Seq2Seq(model_fpath, "t5-base")
    lifted_ltl = lt_module.type_constrained_decode([query])[0]
    return lifted_ltl


if __name__ == "__main__":
    location = "boston"
    ablation = None  # "text", "image", None

    model_fpath = os.path.join(os.path.expanduser("~"), "ground", "models", "checkpoint-best")
    data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
    graph_dpath = os.path.join(data_dpath, "maps", loc2gid[location])
    osm_fpath = os.path.join(data_dpath, "osm", f"{location}.json")
    utt_fpath = os.path.join(data_dpath, f"utts_{location}.txt")
    results_dpath = os.path.join(os.path.expanduser("~"), "ground", "results")
    srer_out_fname = f"srer_outs_{location}.json" if ablation else f"srer_outs_{location}_ablate_{ablation}.json"
    reg_out_fname = srer_out_fname.replace("srer", "reg")
    spg_out_fname = srer_out_fname.replace("srer", "spg")
    topk = 3  # top 3 most likely landmarks grounded by REG


    # Spatial Referring Expression Recognition
    srer_out_fpath = os.path.join(results_dpath, srer_out_fname)
    srer_outs = []
    utts = load_from_file(utt_fpath)
    for utt in tqdm(utts, desc='Performing spatial referring expression recognition (SRER)...'):
        _, rer_out = srer(utt)
        srer_outs.append(rer_out)
    save_to_file(srer_outs, srer_out_fpath)


    # Referring Expression Grounding
    reg(results_dpath, graph_dpath, osm_fpath, srer_out_fname, topk, ablation)


    # Spatial Predicate Grounding
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


    # Lifted Translation
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
