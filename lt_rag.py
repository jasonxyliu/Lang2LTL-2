import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from openai_models import get_embed, translate
from utils import deserialize_props_str, load_from_file, save_to_file


def lt(data_dpath, srer_out_fname, raw_data, topk):
    lt_outs = []
    srer_outs = load_from_file(os.path.join(data_dpath, srer_out_fname))

    for srer_out in srer_outs:
        query = [srer_out['lifted_utt'], json.dumps(list(srer_out["lifted_symbol_map"].keys()))]
        lifted_ltl = lifted_translate(query, raw_data, topk)

        print(f"query: {query}\n{lifted_ltl}\n")

        breakpoint()

    save_to_file(lt_outs, os.path.join(data_dpath, srer_out_fname.replace("srer", "lt")))

    return lifted_ltl



def lifted_translate(query, raw_data, topk):
    prompt_examples = retriever(query, raw_data, topk)

    breakpoint()

    lifted_ltl = translate(query[0], prompt_examples)
    return lifted_ltl


def retriever(query, raw_data, topk):
    nprops_query = len(deserialize_props_str(query[1]))
    query = query[:1]

    # Select lifted commands and formulas with same nprops as query command
    # not work with SRER output for "go to a at most five times"
    # data = []
    # for ltl_type, props, utt, ltl in raw_data:
    #     nprops = len(deserialize_props_str(props))
    #     entry = [ltl_type, props, utt, ltl]
    #     if nprops == nprops_query and entry not in data:
    #         data.append(entry)
    # print(f"{len(data)} templates matched query nprops")
    data = raw_data

    # Embed lifted commands then save or load from cache
    embeds = []
    embeds_fpath = os.path.join(data_dpath, f"data_embeds.pkl")
    utt2embed = load_from_file(embeds_fpath) if os.path.isfile(embeds_fpath) else {}
    embeds_updated = False
    for idx, (_, _, utt, _) in enumerate(data):
        # print(f"{idx}/{len(data)}. getting embedding:\n{utt}")
        if utt in utt2embed:
            embed = utt2embed[utt]
        else:
            embed = get_embed(utt)  # embedding
            utt2embed[utt] = embed
            embeds_updated = True
            print(f"added new embedding\n{utt}")
        embeds.append(embed)
    if embeds_updated:
        save_to_file(utt2embed, embeds_fpath)
    embeds = np.array(embeds)

    # Retrieve prompt examples
    query_embed = get_embed(query)
    query_scores = cosine_similarity(np.array(query_embed).reshape(1, -1), embeds)[0]
    data_sorted = sorted(zip(query_scores, data), reverse=True)

    prompt_examples = []
    for score, (ltl_type, props, utt, ltl) in data_sorted[:topk]:
        # print(score)
        prompt_examples.append(f"Command: \"{utt}\"\nLTL formula: \"{ltl}\"")
        print(f"Command: \"{utt}\"\nLTL formula: \"{ltl}\"\n")

    return prompt_examples


def run_exp_lt_rag(spg_out_fpath, lt_out_fpath, raw_data, topk):
    if not os.path.isfile(lt_out_fpath):
        spg_outs = load_from_file(spg_out_fpath)
        for spg_out in tqdm(spg_outs, desc="Running lifted translation (LT) module (method='rag')"):
            query = [spg_out['lifted_utt'], json.dumps(list(spg_out["lifted_symbol_map"].keys()))]
            print(f"query: {query}\n{lifted_ltl}\n")

            spg_out["lifted_ltl"] = lifted_translate(query, raw_data, topk)

        save_to_file(spg_outs, lt_out_fpath)


if __name__ == "__main__":
    data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
    data_fpath = os.path.join(data_dpath, "symbolic_batch12_noperm.csv")
    raw_data = load_from_file(data_fpath)

    srer_out_fname = "srer_outs_blackstone.json"
    lt(data_dpath, srer_out_fname, raw_data, topk=50)

    # query = ["go to a at most five times", "['a', 'a', 'a', 'a', 'a']"]
    # lifted_translate(query, raw_data, topk=50)
