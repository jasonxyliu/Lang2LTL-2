import os

from srer import srer
from reg import reg
from spg import load_lmks, spg
from lt import Seq2Seq, lt


LOC2GID = {
    "lab": "downloaded_graph_2024-02-02_10-55-35",
    "alley": "downloaded_graph_2024-02-02_14-26-54",
    "blackstone": "downloaded_graph_2024-01-27_07-48-53",
    "boston": "boston",
    "auckland": "auckland",
}  # location to Spot graph ID


def ground(graph_dpath, osm_fpath, model_fpath, utt, ablate, topk, rel_embeds_fpath):
    """
    Grounding API function
    """
    # Spatial Referring Expression Recognition (SRER)
    _, srer_out = srer(utt)  # subsequent module outputs also stored in this dict

    # Referring Expression Grounding (REG)
    reg(graph_dpath, osm_fpath, [srer_out], topk, ablate)

    # Spatial Predicate Grounding (SPG)
    landmarks = load_lmks(graph_dpath, osm_fpath)
    srer_out['spg_results'] = spg(landmarks, srer_out, topk, rel_embeds_fpath)

    # Lifted Translation (LT)
    lt_module = Seq2Seq(model_fpath, "t5-base")
    lt(srer_out, lt_module)

    # Substitute symbols by groundings of spatial referring expressions
    grounded_ltl = srer_out["lifted_ltl"]
    sym2ground = {}
    for symbol, sre in srer_out["lifted_symbol_map"].items():
        grounding = srer_out["spg_results"][sre][0]["target"]
        sym2ground[symbol] = grounding
        grounded_ltl.replace(symbol, grounding)
    srer_out["grounded_ltl"] = grounded_ltl
    srer_out["sym2ground"] = sym2ground

    return srer_out


if __name__ == "__main__":
    location = "lab"
    ablate = "text"  # "text", "image", None
    topk = 5  # top k most likely landmarks grounded by REG

    data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
    graph_dpath = os.path.join(data_dpath, "maps", LOC2GID[location])
    osm_fpath = os.path.join(data_dpath, "osm", f"{location}.json")
    model_fpath = os.path.join(os.path.expanduser("~"), "ground", "models", "checkpoint-best")
    utt_fpath = os.path.join(data_dpath, f"utts_{location}.txt")
    results_dpath = os.path.join(os.path.expanduser("~"), "ground", "results")
    rel_embeds_fpath = os.path.join(os.path.expanduser("~"), "ground", "results", f"known_rel_embeds.json")
    srer_out_fname = f"srer_outs_{location}_ablate_{ablate}.json" if ablate else f"srer_outs_{location}.json"
    reg_out_fname = srer_out_fname.replace("srer", "reg")
    spg_out_fname = srer_out_fname.replace("srer", "spg")

    utts = [
        "go to the couch in front of the TV, the couch to the left of the kitchen counter, the kitchen counter between the couch and the refrigerator, the table next to the door, and the chair on the left of the bookshelf in any order",
    ]
    for idx, utt in enumerate(utts):
        ground_out = ground(graph_dpath, osm_fpath, model_fpath, utt, ablate, topk, rel_embeds_fpath)
        print(f"{idx}/{len(utts)}\nInput utt: {ground_out["grounded_ltl"]}\nOutput LTL: {ground_out}\n")
