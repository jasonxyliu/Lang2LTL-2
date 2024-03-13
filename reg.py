import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from openai_models import GPT4V, get_embed
from utils import load_from_file, save_to_file


def embed_images(img_fpaths, cap_dpath, embed_dpath):
    img_embeds = {}
    for img_fpath in img_fpaths:
        img_id = Path(img_fpath).stem
        cap_fpath = os.path.join(cap_dpath, f"{img_id}.txt")
        embed_fpath = os.path.join(embed_dpath, f"{img_id}.pkl")

        if os.path.isfile(embed_fpath):
            img_cap = load_from_file(cap_fpath)
            img_embed = load_from_file(embed_fpath)
        else:
            img_cap = GPT4V().caption(img_fpath)  # image caption
            save_to_file(img_cap, cap_fpath)
            img_embed = get_embed(img_cap)  # embed image captioin
            save_to_file(img_embed, embed_fpath)

        img_embeds[img_id] = img_embed
    return img_embeds


def embed_texts(txts, embed_dpath):
    txt_embeds = {}
    for lmk_name, txt in txts.items():
        txt_id = lmk_name.lower().replace(" ", "_")
        embed_fpath = os.path.join(embed_dpath, f"{txt_id}.pkl")

        if os.path.isfile(embed_fpath):
            txt_emebed = load_from_file(embed_fpath)
        else:
            txt["name"] = lmk_name  # add landmark name into its textual description
            txt_emebed = get_embed(txt)
            save_to_file(txt_emebed, embed_fpath)

        txt_embeds[lmk_name] = txt_emebed
    return txt_embeds


class REG():
    """
    Referring Expression Grounding (REG) module. Use semantic description of landmarks and objects in text and images.
    """
    def __init__(self, img_embeds, txt_embeds, query_cache_fpath):
        self.sem_ids,sem_embeds = [], []

        if img_embeds:
            self.sem_ids += list(img_embeds.keys())
            sem_embeds += list(img_embeds.values())

        if txt_embeds:
            self.sem_ids += list(txt_embeds.keys())
            sem_embeds += list(txt_embeds.values())

        self.sem_embeds = np.array(sem_embeds)

        if os.path.isfile(query_cache_fpath):
            self.query_cache = load_from_file(query_cache_fpath)
        else:
            self.query_cache = {}
        self.query_cache_fpath = query_cache_fpath

    def query(self, query, topk):
        if query in self.query_cache:
            query_embeds = self.query_cache[query]
        else:
            query_embeds = get_embed(query)
            self.query_cache[query] = query_embeds
            save_to_file(self.query_cache, self.query_cache_fpath)

        query_scores = cosine_similarity(np.array(query_embeds).reshape(1, -1), self.sem_embeds)[0]
        lmks_sorted = sorted(zip(query_scores, self.sem_ids), reverse=True)
        return lmks_sorted[:topk]


def reg(graph_dpath, osm_fpath, srer_outs, topk, ablate, in_cache_fpath):
    img_embeds, txt_embeds = None, None

    if not ablate or ablate == "both" or ablate == "text":
        img_cap_dpath = os.path.join(graph_dpath, "image_captions")
        os.makedirs(img_cap_dpath, exist_ok=True)
        img_embed_dpath = os.path.join(graph_dpath, "image_embeds")
        os.makedirs(img_embed_dpath, exist_ok=True)

        img_dpath = os.path.join(graph_dpath, "images")  # SLAM
        img_fpaths = sorted([os.path.join(img_dpath, fname) for fname in os.listdir(img_dpath) if ".jpg" in fname or ".png" in fname])
        img_embeds = embed_images(img_fpaths, img_cap_dpath, img_embed_dpath)

    if not ablate or  ablate == "both" or ablate == "image":
        txt_embed_dpath = os.path.join(graph_dpath, "text_embeds")
        os.makedirs(txt_embed_dpath, exist_ok=True)

        txts = load_from_file(osm_fpath)  # OSM
        txt_embeds = embed_texts(txts, txt_embed_dpath)

    reg = REG(img_embeds, txt_embeds, in_cache_fpath)

    for srer_out in tqdm(srer_outs, desc="Running referring expression grounding (REG) module"):
        grounded_sre_to_preds = {}

        for sre, spatial_pred in srer_out["sre_to_preds"].items():
            if spatial_pred:  # spatial referring expression
                spatial_relation = list(spatial_pred.keys())[0]
                res =  list(spatial_pred.values())[0]
            else:
                spatial_relation = "None"  # reference expression without spatial relation
                res = [sre]

            grounded_res = []
            for query in enumerate(res):
                lmk_candidates = reg.query(query, topk=topk)
                grounded_res.append(lmk_candidates)
            grounded_sre_to_preds[sre] = {spatial_relation: grounded_res}

        srer_out["grounded_sre_to_preds"] = grounded_sre_to_preds


def run_exp_reg(srer_out_fpath, graph_dpath, osm_fpath, topk, ablate, reg_out_fpath, in_cache_fpath):
    if not os.path.isfile(reg_out_fpath):
        srer_outs = load_from_file(srer_out_fpath)
        reg(graph_dpath, osm_fpath, srer_outs, topk, ablate, in_cache_fpath)
        save_to_file(srer_outs, reg_out_fpath)


if __name__ == "__main__":
    data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
    graph_dpath = os.path.join(data_dpath, "maps", "downloaded_graph_2024-01-27_07-48-53")
    osm_fpath = os.path.join(data_dpath, "osm", "blackstone.json")
    results_dpath = os.path.join(os.path.expanduser("~"), "ground", "results")
    srer_out_fname = "srer_outs_blackstone.json"

    srer_outs = load_from_file(os.path.join(results_dpath, srer_out_fname))
    reg_outs = reg(graph_dpath, osm_fpath, srer_out_fname, topk=5, ablate=None)
    save_to_file(reg_outs, os.path.join(results_dpath, srer_out_fname.replace("srer", "reg")))
