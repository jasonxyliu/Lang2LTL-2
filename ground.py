import os

from srer import srer
from reg import reg
from spg import init, spg
from lt_s2s_sup_tcd import Seq2Seq


LOC2GID = {
    "indoor_env_0": "downloaded_graph_2024-02-02_10-55-35",
    "alley": "downloaded_graph_2024-02-02_14-26-54",
    "blackstone": "downloaded_graph_2024-01-27_07-48-53",
    "boston": "boston",
    "auckland": "auckland",
}  # location to Spot graph ID


def ground(graph_dpath, osm_fpath, model_fpath, utt):
    """
    Grounding API function
    """
    # Spatial Referring Expression Recognition (SRER)
    _, srer_out = srer(utt)

    # Referring Expression Grounding (REG)
    reg_out = reg(graph_dpath, osm_fpath, [srer_out], topk, ablation)[0]

    # Spatial Predicate Grounding (SPG)
    init(graph_dpath, osm_fpath)
    reg_out['spg_results'] = spg(reg_out, topk)

    # Lifted Translation (LT)
    lifted_utt = srer_out["lifted_utt"]
    query = lifted_utt.translate(str.maketrans('', '', ',.'))
    lt_module = Seq2Seq(model_fpath, "t5-base")
    reg_out["lifted_ltl"] = lt_module.type_constrained_decode([query])[0]

    # Substitute symbols by groundings of spatial referring expressions
    grounded_ltl = reg_out["lifted_ltl"]
    for symbol, sre in reg_out["lifted_symbol_map"].items():
        grounding = reg_out["spg_results"][sre][0]["target"]
        grounded_ltl.replace(symbol, grounding)
    reg_out["grounded_ltl"] = grounded_ltl

    return reg_out


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

    utt = "go to the couch in front of the TV, the couch to the left of the kitchen counter, the kitchen counter between the couch and the refrigerator, the table next to the door, and the chair on the left of the bookshelf in any order"
    ground_out = ground(graph_dpath, osm_fpath, model_fpath, utt)
    print(ground_out["grounded_ltl"])
