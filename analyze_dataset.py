"""
Analyze the evaluation dataset.
"""
from utils import load_from_file


def count_lmks(loc, res_fpath, obj_fpath, osm_fpath):
    res = load_from_file(res_fpath)
    objs = load_from_file(obj_fpath)
    osm_lmks = load_from_file(osm_fpath)  # all OSM lmks including distractor lmks

    num_objs = len(objs.keys()) - 1  # robot inital location, waypoint_0
    num_lmks = len(res.keys()) - num_objs
    num_distractor_lmks = len(osm_lmks) - num_lmks
    num_all_lmks = len(osm_lmks) + num_objs

    print(f"Location: {loc}")
    print(f"number of OSM lmks : {num_lmks}")
    print(f"number of objs : {num_objs}")
    print(f"number of distractor OSM lmks: {num_distractor_lmks}")
    print(f"number of all lmks : {num_all_lmks}\n")


def count_sres(locs, lmk_grounds_fpath):
    all_lmk_grounds = load_from_file(lmk_grounds_fpath)
    city2nsres = {loc: 0 for loc in locs}

    for city, lmk_grounds in all_lmk_grounds.items():
        for _, grounds in lmk_grounds.items():
            city2nsres[city] += len(grounds)

    print(f"number of SRES per city: {city2nsres}")
    print(f"Total number of SRES: {sum(city2nsres.values())}")


if __name__ == "__main__":
    locs = ["providence", "auckland", "boston", "san_francisco"]

    for loc in locs:
        res_fpath = f"data/dataset/{loc}/{loc}_res.json"
        obj_fpath = f"data/maps/{loc}/obj_locs.json"
        osm_fpath = f"data/osm/{loc}.json"
        count_lmks(loc, res_fpath, obj_fpath, osm_fpath)

    lmk_grounds_fpath = f"data/dataset/true_lmk_grounds.json"
    count_sres(locs, lmk_grounds_fpath)
