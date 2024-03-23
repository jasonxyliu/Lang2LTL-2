import os
import argparse

from srer import srer
from reg import reg
from spg import load_lmks, spg
from lt import Seq2Seq, lt
from utils import load_from_file, save_to_file


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
    sym2ground = {}  # only language grounding: language grounding symbol to lmk ID. robot demo: language grounding symbol to planner symbol
    for symbol, sre in srer_out["lifted_symbol_map"].items():
        ground = srer_out["grounded_sps"][sre][0]["target"]
        sym2ground[symbol] = lmk2sym[ground] if lmk2sym else ground
    srer_out["sym2ground"] = sym2ground

    #  Robot demo only; replace language grounding symbol by planner symbol
    if lmk2sym:
        grounded_ltl = srer_out["lifted_ltl"]
        for ground_sym in sym2ground.keys():
            grounded_ltl = grounded_ltl.replace(ground_sym, f"<{ground_sym}>")
        for ground_sym, plan_sym in sym2ground.items():
            grounded_ltl = grounded_ltl.replace(f"<{ground_sym}>", plan_sym)
        srer_out["grounded_ltl"] = grounded_ltl

    return srer_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc", type=str, default="outdoor", choices=["indoor", "outdoor"], help="env name.")
    parser.add_argument("--ablate", type=str, default=None, choices=["both", "image", "text", None], help="ablate out a modality (indoor: text. outdoor: None).")
    parser.add_argument("--topk", type=int, default=10, help="top k most likely landmarks grounded by REG.")
    args = parser.parse_args()

    data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
    graph_dpath = os.path.join(data_dpath, "maps", args.loc)
    lmk2sym_fpath = os.path.join(graph_dpath, "lmk2sym.json")
    lmk2sym = load_from_file(lmk2sym_fpath) if os.path.isfile(lmk2sym_fpath) else {}  # landmark ID to planner symbol used for robot demo
    osm_fpath = os.path.join(data_dpath, "osm", f"{args.loc}.json")
    model_fpath = os.path.join(os.path.expanduser("~"), "ground", "models", "checkpoint-best")
    rel_embeds_fpath = os.path.join(data_dpath, f"known_rel_embeds.json")
    reg_in_cache_fpath = os.path.join(data_dpath, f"reg_in_cache_{args.loc}.pkl")
    utt_fpath = os.path.join(data_dpath, f"utts_{args.loc}.txt")
    results_dpath = os.path.join(os.path.expanduser("~"), "ground", "results_spot", args.loc)
    os.makedirs(results_dpath, exist_ok=True)
    out_fpath = os.path.join(results_dpath, "srer_outs.json")

    utts = [
        # "go to the couch in front of the TV, the couch to the left of the kitchen counter, the kitchen counter between the couch and the refrigerator, the table next to the door, and the chair on the left of the bookshelf in any order",

        "Visit the white car, then go to the red brick wall and then go to the silver car near the apartment, in addition you can never go to the apartment once you've seen the white car"
    ]

    ground_outs = []
    for idx, utt in enumerate(utts):
        ground_out = ground(graph_dpath, lmk2sym, osm_fpath, model_fpath, utt, args.ablate, args.topk, rel_embeds_fpath, reg_in_cache_fpath)
        print(f"***** {idx}/{len(utts)}\nInput utt: {utt}\nLifted LTL: {ground_out['lifted_ltl']}\nSymbol to Grounding: {ground_out['sym2ground']}")
        if lmk2sym:
            print(f"Grounded LTL: {ground_out['grounded_ltl']}")
        ground_outs.append(ground_out)
    save_to_file(ground_outs, out_fpath)
