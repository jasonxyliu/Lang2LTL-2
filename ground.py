import os

from srer import srer
from reg import reg
from spg import load_lmks, spg
from lt import Seq2Seq, lt
from utils import load_from_file, save_to_file


LOC2GID = {
    "indoor": "downloaded_graph_2024-02-02_10-55-35",
    "outdoor": "outdoor",
    "providence": "providence",  # "downloaded_graph_2024-01-27_07-48-53"
    "boston": "boston",
    "auckland": "auckland",
    "san_francisco": "san_francisco",
}  # location to Spot graph ID


def ground(graph_dpath, lmk2sym, osm_fpath, model_fpath, utt, ablate, topk, rel_embeds_fpath, reg_in_cache_fpath):
    """
    Grounding API function
    """
    # Spatial Referring Expression Recognition (SRER)
    _, srer_out = srer(utt)  # subsequent module outputs also stored in this dict

    # Referring Expression Grounding (REG)
    reg(graph_dpath, osm_fpath, [srer_out], topk, ablate, reg_in_cache_fpath)

    # Spatial Predicate Grounding (SPG)
    landmarks = load_lmks(graph_dpath, osm_fpath)
    srer_out["grounded_sps"] = spg(landmarks, srer_out, topk, rel_embeds_fpath)

    # Lifted Translation (LT)
    lt_module = Seq2Seq(model_fpath, "t5-base")
    lt(srer_out, lt_module)

    # Substitute symbols by groundings of spatial referring expressions
    sym2ground = {}
    if lmk2sym:
        sym2sym = {}
    for symbol, sre in srer_out["lifted_symbol_map"].items():
        ground = srer_out["grounded_sps"][sre][0]["target"]
        sym2ground[symbol] = ground
        if lmk2sym:
            sym2sym[symbol] = lmk2sym[ground]  # language grounding symbol to planner symbol
    srer_out["sym2ground"] = sym2ground
    if lmk2sym:
        srer_out["sym2sym"] = sym2sym

    if lmk2sym:
        grounded_ltl = srer_out["lifted_ltl"]
        for ground_sym in sym2sym.keys():
            grounded_ltl = grounded_ltl.replace(ground_sym, f"<{ground_sym}>")
        for ground_sym, plan_sym in sym2sym.items():
            grounded_ltl = grounded_ltl.replace(f"<{ground_sym}>", plan_sym)
        srer_out["grounded_ltl"] = grounded_ltl

    return srer_out


if __name__ == "__main__":
    loc = "outdoor"
    ablate = None  # "text", "image", None
    # loc = "indoor"
    # ablate = "text"  # "text", "image", None
    topk = 10  # top k most likely landmarks grounded by REG

    data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
    graph_dpath = os.path.join(data_dpath, "maps", LOC2GID[loc])
    lmk2sym_fpath = os.path.join(graph_dpath, "lmk2sym.json")
    lmk2sym = load_from_file(lmk2sym_fpath) if os.path.isfile(lmk2sym_fpath) else {}  # landmark ID to planner symbol used for robot demo
    osm_fpath = os.path.join(data_dpath, "osm", f"{loc}.json")
    model_fpath = os.path.join(os.path.expanduser("~"), "ground", "models", "checkpoint-best")
    rel_embeds_fpath = os.path.join(data_dpath, f"known_rel_embeds.json")
    reg_in_cache_fpath = os.path.join(data_dpath, f"reg_in_cache_{loc}.pkl")
    utt_fpath = os.path.join(data_dpath, f"utts_{loc}.txt")
    results_dpath = os.path.join(os.path.expanduser("~"), "ground", "results_spot", loc)
    os.makedirs(results_dpath, exist_ok=True)
    out_fpath = os.path.join(results_dpath, "srer_outs.json")

    utts = [
        # "go to the couch in front of the TV, the couch to the left of the kitchen counter, the kitchen counter between the couch and the refrigerator, the table next to the door, and the chair on the left of the bookshelf in any order",

        "Visit the white car, then go to the red brick wall and then go to the silver car near the apartment, in addition you can never go to the apartment once you've seen the white car"
    ]
    ground_outs = []
    for idx, utt in enumerate(utts):
        ground_out = ground(graph_dpath, lmk2sym, osm_fpath, model_fpath, utt, ablate, topk, rel_embeds_fpath, reg_in_cache_fpath)
        if lmk2sym:
            print(f"***** {idx}/{len(utts)}\nInput utt: {utt}\nLifted LTL: {ground_out['lifted_ltl']}\n{ground_out['sym2sym']}\n{ground_out['grounded_ltl']}")
        else:
            print(f"***** {idx}/{len(utts)}\nInput utt: {utt}\nLifted LTL: {ground_out['lifted_ltl']}\n{ground_out['sym2ground']}")
        ground_outs.append(ground_out)
    save_to_file(ground_outs, out_fpath)
