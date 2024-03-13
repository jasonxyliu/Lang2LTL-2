import os
import argparse
import logging
from collections import defaultdict
import itertools
import random
import re

from ground import LOC2GID
from utils import load_from_file, save_to_file


def split_true_lmk_grounds(loc, lmks_fpath, obj_fpath, sp_fpath, res_fpath):
    """
    Split ``true_lmk_grounds.json`` into two files contains referring expressions per landmark
    and grounded spatial predications per spatial relation for each location.
    """
    lmk_grounds = load_from_file(lmks_fpath)[loc]
    obj_ids = [obj_id for obj_id in load_from_file(obj_fpath).keys() if obj_id != "waypoint_0"]  # landmarks described by image
    lmk_ids = [lmk_id for lmk_id in lmk_grounds.keys() if lmk_id not in obj_ids]  # landmarks described by text

    sp_grounds = {"text": defaultdict(list), "image": defaultdict(list), "both": defaultdict(list)}
    res = defaultdict(lambda: defaultdict(list))

    for lmk, grounds in lmk_grounds.items():
        for ground in grounds:
            if "*" in ground:  # unique referring expression can identify landmark without anchor
                modality = "text" if lmk in lmk_ids else "image"
                res[lmk]["proper_names"].append(ground["*"])
                if lmk not in sp_grounds[modality]["None"]:
                    sp_grounds[modality]["None"].append(lmk)
            elif "@" in ground:  # ambiguous referring expression if used without anchor
                res[lmk]["generic_names"].append(ground["@"])
            else:  # referring expression grounding
                rel = list(ground.keys())[0]

                if len(ground[rel]) == 2:
                    if ground[rel][0][0] in lmk_ids:
                        if ground[rel][1][0] in lmk_ids:
                            modality = "text"
                        else:
                            modality = "both"
                    else:
                        if ground[rel][1][0] in lmk_ids:
                            modality = "both"
                        else:
                            modality = "image"
                elif len(ground[rel]) == 3:
                    if ground[rel][0][0] in lmk_ids:
                        if ground[rel][1][0] in lmk_ids:
                            if ground[rel][2][0] in lmk_ids:
                                modality = "text"
                            else:
                                modality = "both"
                        else:
                            modality = "both"
                    else:
                        if ground[rel][1][0] in lmk_ids:
                            modality = "both"
                        else:
                            if ground[rel][2][0] in lmk_ids:
                                modality = "image"
                            else:
                                modality = "both"
                else:
                    raise IndexError(f"Incorrect number of RE grounds (must be 2 or 3): {ground[rel]}")

                if ground[rel] not in sp_grounds[modality][rel]:
                    sp_grounds[modality][rel].append(ground[rel])

    save_to_file(sp_grounds, sp_fpath)
    save_to_file(res, res_fpath)


def construct_dataset(ltl_fpath, obj_fpath, osm_fpath, sp_fpath, res_fpath, utts_fpath, outs_fpath, nsamples, seed):
    """
    Generate input utterances and ground truth results for each grounding module.
    """
    random.seed(seed)

    lifted_data = load_from_file(ltl_fpath)
    sp_grounds_all = load_from_file(sp_fpath)
    res_all = load_from_file(res_fpath)

    ltl2data = defaultdict(set)
    utts = []
    for pattern_type, props, utt_lifted, ltl_lifted in lifted_data:
        if utt_lifted not in utts:
            ltl2data[ltl_lifted].add((pattern_type, props, utt_lifted))
    ltl2data = sorted(ltl2data.items(), key=lambda kv: len(kv[0]))

    logging.info(f"# unique lifted LTL formulas: {len(ltl2data)}")
    nutts = 0
    for ltl, data in ltl2data:
        nutts += len(data)
        logging.info(f"{ltl}: {len(data)}")
    logging.info(f"# unique utterances: {nutts}")
    modality2ngrounds = {modality: len(grounds) for modality, grounds in sp_grounds_all.items()}
    logging.info(f"# unique spatial relations: {modality2ngrounds}")
    logging.info(f"# unique landmarks: {len(res_all)}")

    utts = ""
    true_outs = []

    for ltl_lifted, ltl_data in ltl2data:  # every lifted LTL formula
        data_sampled = random.sample(sorted(ltl_data), nsamples) if nsamples else sorted(ltl_data)
        ratio = len(data_sampled) // 3  # text only, image only, both modality

        for data in data_sampled[: ratio]:
            modality = "text"
            utts = construct_utt(modality, data, sp_grounds_all[modality], res_all, utts, true_outs, ltl_lifted)
        for data in data_sampled[ratio: ratio + ratio]:
            modality = "image"
            utts = construct_utt(modality, data, sp_grounds_all[modality], res_all, utts, true_outs, ltl_lifted)
        for data in data_sampled[ratio + ratio:]:
            modality = "both"
            utts = construct_utt(modality, data, sp_grounds_all[modality], res_all, utts, true_outs, ltl_lifted)

    save_to_file(utts, utts_fpath)
    save_to_file(true_outs, outs_fpath)
    logging.info(f"# data points: {len(true_outs)}")


def construct_utt(modality, data, sp_grounds_all, res_all, utts, true_outs, ltl_lifted):
    pattern_type, props_full_str, utt_lifted = data
    props_full = eval(props_full_str)
    props = [props_full[0]] if len(set(props_full)) == 1 else props_full  # e.g., visit a at most twice, ['a', 'a']
    rels = random.sample(sorted(sp_grounds_all), len(props))

    sre_to_preds = {}
    grounded_sre_to_preds = defaultdict(dict)
    grounded_sps = defaultdict(list)

    for rel in rels:  # every sampled spatial relations
        sp_grounds_sampled = random.sample(sp_grounds_all[rel], 1)[0]
        res_true = []

        if rel == "None":  # referring expression without spatial relation
            sre = random.sample(res_all[sp_grounds_sampled]["proper_names"], 1)[0]
            res_true.append(sre)
            sp_true = {"target": sp_grounds_sampled}
        elif len(sp_grounds_sampled) == 1:  # sre with only an anchor
            while "proper_names" not in res_all[sp_grounds_sampled[0]]:
                sp_grounds_sampled = random.sample(sp_grounds_all[rel], 1)[0]
            re_tar = random.sample(res_all[sp_grounds_sampled[0]]["proper_names"], 1)[0]
            res_true.append(re_tar)
            sre = f"{rel} {re_tar}"
            sp_true = {"anchor": [sp_grounds_sampled[0]]}
        else:  # for sre with target and one or two anchors, both proper and generic names are valid
            while "proper_names" not in res_all[sp_grounds_sampled[1][0]] \
                or (len(sp_grounds_sampled) == 3 and "proper_names" not in res_all[sp_grounds_sampled[2][0]]):
                sp_grounds_sampled = random.sample(sp_grounds_all[rel], 1)[0]
            res_tar = list(itertools.chain.from_iterable(res_all[sp_grounds_sampled[0][0]].values()))
            re_tar = random.sample(res_tar, 1)[0]  # target referring expression
            res_true.append(re_tar)
            re_anc1 = random.sample(res_all[sp_grounds_sampled[1][0]]["proper_names"], 1)[0]  # anchor 1 referring expression
            res_true.append(re_anc1)
            if len(sp_grounds_sampled) == 2:
                sre = f"{re_tar} {rel} {re_anc1}"
                sp_true = {"target": sp_grounds_sampled[0][0], "anchor": [sp_grounds_sampled[1][0]]}
            else:
                re_anc2 = random.sample(res_all[sp_grounds_sampled[2][0]]["proper_names"], 1)[0] # anchor 2 referring expression
                res_true.append(re_anc2)
                sre = f"{re_tar} {rel} {re_anc1} and {re_anc2}"
                sp_true = {"target": sp_grounds_sampled[0][0], "anchor": [sp_grounds_sampled[1][0], sp_grounds_sampled[2][0]]}

        sre_to_preds[sre] = {rel: res_true}

        if rel == "None":
            grounded_sre_to_preds[sre][rel] = [[[1.0, sp_grounds_sampled]]]
        else:
            grounded_sre_to_preds[sre][rel] = [[score_ground] for score_ground in [[1.0, sp_ground[0]] for sp_ground in sp_grounds_sampled]]

        grounded_sps[sre].append(sp_true)

    if not utt_lifted.startswith('.'):
        utt_ground = '.' + utt_lifted
    if not utt_ground.endswith('.'):
        utt_ground += '.'
    for prop, sre in zip(props, sre_to_preds.keys()):
        utt_ground = re.sub(rf"(\b)([{prop}])(\W)", rf'\1{sre}\3', utt_ground)
    utt_ground = utt_ground[1:-1]
    utts += f"{utt_ground}\n"

    true_outs.append({
        "pattern_type": pattern_type,
        "modality": modality,
        "utt": utt_ground.strip(),
        "lifted_utt": utt_lifted,
        "props": props_full,
        "sre_to_preds": sre_to_preds,
        "grounded_sre_to_preds": grounded_sre_to_preds,
        "grounded_sps": grounded_sps,
        "lifted_ltl": ltl_lifted
    })
    return utts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc", type=str, default="providence", choices=["providence", "boston", "auckland", "san_francisco"], help="domain name.")
    parser.add_argument("--nsamples", type=int, default=3, help="number of sample utts per LTL formula or None for all.")
    parser.add_argument("--seed", type=int, default=111, help="seed to random sampler.")  # 111 (resreved for ablate)
    args = parser.parse_args()
    loc_id = f"{args.loc}_n{args.nsamples}_seed{args.seed}" if args.nsamples else f"{args.loc}_all_seed{args.seed}"

    data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
    obj_fpath = os.path.join(data_dpath, "maps", f"{LOC2GID[args.loc]}_ablate", "obj_locs.json")
    osm_fpath = os.path.join(data_dpath, "osm_ablate", f"{args.loc}.json")
    dataset_dpath = os.path.join(os.path.expanduser("~"), "ground", "data", "dataset")
    loc_dpath = os.path.join(dataset_dpath, f"{args.loc}_ablate")
    os.makedirs(loc_dpath, exist_ok=True)
    ltl_fpath = os.path.join(dataset_dpath, "ltl_samples_sorted.csv")
    sp_fpath = os.path.join(loc_dpath, f"{args.loc}_sp_grounds.json")
    res_fpath = os.path.join(loc_dpath, f"{args.loc}_res.json")
    utts_fpath = os.path.join(loc_dpath, f"{loc_id}_utts.txt")
    outs_fpath = os.path.join(loc_dpath, f"{loc_id}_true_results.json")

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(loc_dpath, f"{args.loc}_synthetic_dataset.log"), mode='w'),
                            logging.StreamHandler()
                        ]
    )
    logging.info(f"Generating dataset location: {args.loc}\n***** Dataset Statisitcs\n")

    # obj_ids = sorted([Path(fname).stem for fname in os.listdir(img_dpath) if ".png" in fname or ".jpg" in fname])

    # if not os.path.isfile(sp_fpath) or not os.path.isfile(res_fpath):
    lmks_fpath = os.path.join(dataset_dpath, f"true_lmk_grounds_ablate.json")
    split_true_lmk_grounds(args.loc, lmks_fpath, obj_fpath, sp_fpath, res_fpath)

    # if not os.path.isfile(utts_fpath) or not os.path.isfile(outs_fpath):
    construct_dataset(ltl_fpath, obj_fpath, osm_fpath, sp_fpath, res_fpath, utts_fpath, outs_fpath, args.nsamples, args.seed)
