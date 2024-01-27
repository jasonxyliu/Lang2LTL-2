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
        self.img_ids = list(self.img_embeds.keys())
        self.txt_embeds = np.array(list(txt_embeds.values()))
        self.txt_ids = list(self.txt_embeds.keys())
        self.txt_embed_model = TextEmbedding()

    def query(self, query, topk):
        query_embeds = self.txt_embed_model.embed(query)
        query_scores = cosine_similarity(np.array(query_embeds).reshape(1, -1), np.concatenate((self.img_embeds, self.txt_embeds), axis=0))[0]

        lmks_sorted = sorted(zip(query_scores, self.img_ids+self.txt_ids), reverse=True)

        for score, sem_id in lmks_sorted:
            print(f"{sem_id}: {score}")

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


if __name__ == "__main__":
    data_dpath = os.path.join(os.path.expanduser("~"), "lang2ltl", "data")
    graph_dpath = os.path.join(data_dpath, "maps", "downloaded_graph_2024-01-16_13-40-18")

    img_embed_dpath = os.path.join(graph_dpath, "image_embeds")
    img_cap_dpath = os.path.join(img_embed_dpath, "captions")
    os.makedirs(img_cap_dpath, exist_ok=True)
    img_embed_dpath = os.path.join(img_embed_dpath, "embeddings")
    os.makedirs(img_embed_dpath, exist_ok=True)

    txt_embed_dpath = os.path.join(data_dpath, "text_embeds")
    os.makedirs(txt_embed_dpath, exist_ok=True)

    img_dpath = os.path.join(graph_dpath, "images")  # SLAM
    img_fpaths = sorted([os.path.join(img_dpath, fname) for fname in os.listdir(img_dpath) if "jpg" in fname])
    img_embeds = embed_images(img_fpaths, img_cap_dpath, img_embed_dpath)

    txts = load_from_file(os.path.join(data_dpath, "osm", "osm_temp.json"))  # OSM
    txt_embeds = embed_texts(txts, txt_embed_dpath)

    queries = ["bookshelf", "desk and chair", "kitchen cabinet", "blue couch", "red couch", "refrigerator", "door", "white board", "TV"]

    for idx, query in enumerate(queries):
        print(f"{idx}: {query}")
        reg = REG(img_embeds, txt_embeds)
        # reg = REG(np.array(list(img_embeds.values())), np.array(list(txt_embeds.values())))
        lmk_candidates = reg.query(query, topk=5)

        print("\n\n")

    # logging.basicConfig(level=logging.DEBUG,
    #                         format='%(message)s',
    #                         handlers=[
    #                             logging.FileHandler(f'exp.log', mode='w'),
    #                             logging.StreamHandler()
    #                         ]
    # )
