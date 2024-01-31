import os
from tqdm import tqdm

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
    model_fpath = os.path.join(os.path.expanduser("~"), "ground", "models", "checkpoint-best")
    data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
    graph_dpath = os.path.join(data_dpath, "maps", "downloaded_graph_2024-01-27_07-48-53")
    osm_fpath = os.path.join(data_dpath, "osm", "blackstone.json")
    utt_fpath = os.path.join(data_dpath, "utts_blackstone.txt")
    srer_out_fname = "srer_outs_blackstone.json"
    reg_out_fname = srer_out_fname.replace("srer", "reg")
    spg_out_fname = srer_out_fname.replace("srer", "spg")


    # Spatial Referring Expression Recognition
    rer_outs = []
    utts = list(filter(None, [X.strip() for X in open(utt_fpath, 'r').readlines()]))  # load commands, filter empty strings

    for utt in tqdm(utts, desc='Performing referring expression recognition (RER)...'):
        # -- extract RER info from LLM interaction:
        _, rer_out = srer(utt)
        rer_outs.append(rer_out)

    save_to_file(rer_outs, os.path.join(data_dpath, srer_out_fname))


    # Referring Expression Grounding
    reg(data_dpath, graph_dpath, osm_fpath, srer_out_fname, topk=3)


    # Spatial Predicate Grounding
    reg_outs = load_from_file(os.path.join(data_dpath, reg_out_fname))
    init(osm_landmark_file=osm_fpath)

    spg_outs = []
    for reg_out in reg_outs:
        # -- make a copy of the dictionary for a corresponding utterance:
        spg_out = reg_out

        # -- add a new field to the dictionary with the final output of SPG (if any):
        spg_out['spg_results'] = spg(reg_out, topk=3)
        spg_out.pop('grounded_spatial_preds')
        spg_outs.append(spg_out)            

    save_to_file(spg_outs, os.path.join(data_dpath, srer_out_fname.replace("srer", "spg")))


    # Lifted Translation
    srer_outs = load_from_file(os.path.join(data_dpath, srer_out_fname))
    lt_outs = []

    for srer_out in srer_outs:
        lifted_utt = srer_out["lifted_utt"]
        lifted_ltl = ground(lifted_utt, model_fpath)

        srer_out["lifted_ltl"] = lifted_ltl
        lt_outs.append(srer_out)

        print(f"{lifted_utt}\n{lifted_ltl}\n")
        # breakpoint()

    save_to_file(lt_outs, os.path.join(data_dpath, srer_out_fname.replace("srer", "lt")))

    # lifted_utt = "go to a at most five times"
    # lifted_ltl = ground(lifted_utt, model_fpath)
    # print(lifted_ltl)
