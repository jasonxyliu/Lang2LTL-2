import os
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from openai_models import GPT4V, TextEmbedding
from utils import load_from_file, save_to_file


class REG():
    """
    Referring Expression Grounding (REG) module.
    Use semantic description of landmarks and objects in text and images.
    """
    def __init__(self, img_embeds, txt_embeds):
        self.img_embeds = np.array(list(img_embeds.values()))
        self.img_ids = list(img_embeds.keys())
        self.txt_embeds = np.array(list(txt_embeds.values()))
        self.txt_ids = list(txt_embeds.keys())
        self.txt_embed_model = TextEmbedding()

    def query(self, query, topk):
        query_embeds = self.txt_embed_model.embed(query)
        query_scores = cosine_similarity(np.array(query_embeds).reshape(1, -1), np.concatenate((self.img_embeds, self.txt_embeds), axis=0))[0]

        lmks_sorted = sorted(zip(query_scores, self.img_ids+self.txt_ids), reverse=True)

        # for score, sem_id in lmks_sorted:
        #     print(f"{sem_id}: {score}")

        return lmks_sorted[:topk]


def embed_images(img_fpaths, cap_dpath, embed_dpath):
    img_embeds = {}
    for img_fpath in img_fpaths:
        img_id = Path(img_fpath).stem
        cap_fpath = os.path.join(cap_dpath, f"{img_id}.pkl")
        embed_fpath = os.path.join(embed_dpath, f"{img_id}.pkl")

        if os.path.isfile(embed_fpath):
            img_cap = load_from_file(cap_fpath)
            img_embed = load_from_file(embed_fpath)
        else:
            img_cap = GPT4V().caption(img_fpath)
            save_to_file(img_cap, cap_fpath)
            save_to_file(img_cap, os.path.join(cap_dpath, f"{img_id}.txt"))

            img_embed = TextEmbedding().embed(img_cap)
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
            txt_emebed = TextEmbedding().embed(txt)
            save_to_file(txt_emebed, embed_fpath)

        txt_embeds[txt_id] = txt_emebed
    return txt_embeds


def reg(data_dpath, graph_dpath, osm_fpath, srer_out_fname, topk):
    img_cap_dpath = os.path.join(graph_dpath, "image_captions")
    os.makedirs(img_cap_dpath, exist_ok=True)
    img_embed_dpath = os.path.join(graph_dpath, "image_embeds")
    os.makedirs(img_embed_dpath, exist_ok=True)
    txt_embed_dpath = os.path.join(graph_dpath, "text_embeds")
    os.makedirs(txt_embed_dpath, exist_ok=True)

    img_dpath = os.path.join(graph_dpath, "images")  # SLAM
    img_fpaths = sorted([os.path.join(img_dpath, fname) for fname in os.listdir(img_dpath) if "jpg" in fname])
    img_embeds = embed_images(img_fpaths, img_cap_dpath, img_embed_dpath)

    txts = load_from_file(osm_fpath)  # OSM
    txt_embeds = embed_texts(txts, txt_embed_dpath)

    reg = REG(img_embeds, txt_embeds)

    reg_outs = []
    srer_outs = load_from_file(os.path.join(data_dpath, srer_out_fname))

    for srer_out in srer_outs:
        print(f"command: {srer_out['utt']}")
        grounded_sre_to_preds = {}

        for sre, spatial_pred in srer_out["sre_to_preds"].items():
            if spatial_pred:
                spatil_relation = list(spatial_pred.keys())[0]
                res =  list(spatial_pred.values())[0]
            else:
                spatil_relation = "None"  # reference expression w/o spatial relation
                res = [sre]

            grounded_res = []
            for idx, query in enumerate(res):
                lmk_candidates = reg.query(query, topk=topk)
                grounded_res.append(lmk_candidates)
                print(f"{idx}: {sre}\n{query}\n{lmk_candidates}\n")

            grounded_sre_to_preds[sre] = {spatil_relation: grounded_res}

        srer_out["grounded_sre_to_preds"] = grounded_sre_to_preds

        reg_outs.append(srer_out)

    breakpoint()
    save_to_file(reg_outs, os.path.join(data_dpath, srer_out_fname.replace("srer", "reg")))

    # queries = ["bookshelf", "desk and chair", "kitchen cabinet", "blue couch", "red couch", "refrigerator", "door", "white board", "TV"]
    # for idx, query in enumerate(queries):
    #     print(f"{idx}: {query}")
    #     reg = REG(img_embeds, txt_embeds)
    #     # reg = REG(np.array(list(img_embeds.values())), np.array(list(txt_embeds.values())))
    #     lmk_candidates = reg.query(query, topk=5)

    #     print("\n\n")


if __name__ == "__main__":
    data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
    graph_dpath = os.path.join(data_dpath, "maps", "downloaded_graph_2024-01-27_07-48-53")
    osm_fpath = os.path.join(data_dpath, "osm", "blackstone.json")
    srer_out_fname = "srer_outs_blackstone.json"

    reg(data_dpath, graph_dpath, osm_fpath, srer_out_fname, topk=3)
