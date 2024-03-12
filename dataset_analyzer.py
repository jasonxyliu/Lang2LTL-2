from ground import LOC2GID
from utils import load_from_file


def count_lmks(loc):
    res_fpath = f"data/dataset/{loc}/{loc}_res.json"
    obj_fpath = f"data/maps/{LOC2GID[loc]}/obj_locs.json"
    osm_fpath = f"data/osm/{loc}.json"

    res = load_from_file(res_fpath)
    objs = load_from_file(obj_fpath)
    osm_lmks = load_from_file(osm_fpath)  # all OSM lmks including distractor lmks

    num_objs = len(objs.keys()) - 1  # robot inital location, waypoint_0
    num_lmks = len(res.keys()) - num_objs
    num_distractor_lmks = len(osm_lmks) - num_lmks
    num_all_lmks = len(osm_lmks) + num_objs

    print(f"Location: {loc}")
    print(f"# OSM lmks : {num_lmks}")
    print(f"# objs : {num_objs}")
    print(f"# distractor OSM lmks: {num_distractor_lmks}")
    print(f"# all lmks : {num_all_lmks}")


if __name__ == "__main__":
    loc = "san_francisco"
    count_lmks(loc)
