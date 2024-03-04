import os
from pathlib import Path
from tqdm import tqdm
from itertools import product
import numpy as np
import utm
from pyproj import Transformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from load_map import load_map, extract_waypoints
from openai_models import get_embed
from utils import load_from_file, save_to_file


KNOWN_RELATIONS = [
    "left", "right",
    "in front of", "opposite to", "behind",
    "near", "next to", "adjacent to", "close to", "by",
    "between",
    "north of", "south of", "east of", "west of", "northeast of", "northwest of", "southeast of", "southwest of"
]
MAX_RANGE = 60.0  # assume target within this radius of the anchor
DIST_TO_ANCHOR = 2.0  # distance to robot when compute a target location for SRE with only an anchor


def plot_landmarks(landmarks=None, osm_fpth=None):
    """
    Plot landmarks in the shared world space local to the Spot's map.
    """
    plt.figure()
    plt.rcParams.update({'font.size': 5})

    if landmarks:
        plt.scatter(x=[landmarks[L]["x"] for L in landmarks], y=[landmarks[L]["y"] for L in landmarks], c="green", label="landmarks")
        for L in landmarks:
            if "osm_name" not in landmarks[L] and L != "robot":
                plt.text(landmarks[L]["x"], landmarks[L]["y"], L)

    plt.scatter(x=landmarks["robot"]["x"],
                y=landmarks["robot"]["y"], c="orange", label="robot")
    plt.text(landmarks["robot"]["x"],
             landmarks["robot"]["y"], "robot")
    plt.legend()

    if osm_fpth:
        location_name = os.path.splitext(os.path.basename(osm_fpth))[0]
        plt.title(f"Landmark Map: {location_name}")
        plt.show(block=False)
    else:
        plt.title(f"Landmark Map")
        plt.show(block=True)
    plt.savefig(f"{os.path.join(os.path.dirname(osm_fpth), f'{location_name}_landmarks.png')}", dpi=300)
    plt.rcdefaults()  # reset font size to default


def rotate(vec, angle):
    # https://motion.cs.illinois.edu/RoboticSystems/CoordinateTransformations.html
    mat_rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.dot(mat_rot, vec)


def align_coordinates(graph_dpath, waypoints, osm_landmarks, coord_alignment, crs):
    # rotation and translation to align Spot to world Cartesian frame (default: 0, not needed if no Spot graph)
    rotation, translation = 0, 0

    if coord_alignment:
        # If use Spot graph, alignment landmark value is not None, then compute rotation and translation for alignment
        print(" >> Computing alignment from robot to world frame...")
        known_landmark_1 = np.array(crs.transform(coord_alignment[0]["long"], coord_alignment[0]["lat"], 0, radians=False)[:-1])
        known_landmark_2 = np.array(crs.transform(coord_alignment[1]["long"], coord_alignment[1]["lat"], 0, radians=False)[:-1])

        known_waypoint_1 = np.array([waypoints[coord_alignment[0]["waypoint"]]["position"]["x"],
                                     waypoints[coord_alignment[0]["waypoint"]]["position"]["y"]])
        known_waypoint_2 = np.array([waypoints[coord_alignment[1]["waypoint"]]["position"]["x"],
                                     waypoints[coord_alignment[1]["waypoint"]]["position"]["y"]])

        # Use the vector from known landmarks to compute rotation
        vec_lmk_1_to_2 = known_landmark_2 - known_landmark_1
        vec_wp_1_to_2 = known_waypoint_2 - known_waypoint_1

        # Compute the rotation between the known landmark and waypoint
        dir_world = np.arctan2(vec_lmk_1_to_2[1], vec_lmk_1_to_2[0])  # i.e., world coordinate
        dir_robot = np.arctan2(vec_wp_1_to_2[1], vec_wp_1_to_2[0])  # i.e., Spot coordinate

        rotation = dir_world - dir_robot

    landmarks = {}

    if graph_dpath and waypoints:
        # Each image is named after the Spot waypoint ID (auto-generated by GraphNav)
        waypoint_ids = [Path(image_fpath).stem for image_fpath in os.listdir(os.path.join(graph_dpath, "images"))]

        for wid, wp_desc in waypoints.items():
            # NOTE: all landmarks are either one of the following:
            #  1. waypoint_0: robot start location when using GraphNav
            #  2. landmarks whose waypoints created by GraphNav; they have images
            is_landmark = True if wid in waypoint_ids or wp_desc["name"] == "waypoint_0" else False

            if is_landmark:
                cartesian_coords = np.array([wp_desc["position"]["x"], wp_desc["position"]["y"]])

                # Align the Spot's cartesian coordinates to the world frame:
                cartesian_coords = rotate(cartesian_coords, rotation)

                if coord_alignment:
                    # Use the newly rotated point to figure out the translation
                    if wid == coord_alignment[0]["waypoint"]:
                        known_waypoint_1 = cartesian_coords
                    elif wid == coord_alignment[1]["waypoint"]:
                        known_waypoint_2 = cartesian_coords

                lmk_id = "robot" if wp_desc["name"] == "waypoint_0" else wid
                landmarks[lmk_id] = {"x": cartesian_coords[0], "y": cartesian_coords[1]}

        if coord_alignment:
            # Compute translation to align the known landmark from world to Spot space AFTER rotation
            translation = ((known_waypoint_1 - known_landmark_1) + (known_waypoint_2 - known_landmark_2)) / 2.0
    else:
        # This means we are only working with OSM landmarks
        print(" >> WARNING: not using Spot graph")

    # Process then add OSM landmarks if provided
    if osm_landmarks:
        for lmk, lmk_desc in osm_landmarks.items():
            # lmk_id = lmk.lower().replace(' ', '_')

            if "wid" in lmk_desc:
                # OSM landmarks visited by Spot GraphNav have waypoint IDs, just use Spot graph coorindates
                wid = lmk_desc["wid"]
                lmk_cartesian = np.array([landmarks[wid]["x"], landmarks[wid]["y"]])
                landmarks[wid]["osm_name"] = lmk
            else:
                # Convert landmark location to Cartesian coordinate then add computed translation
                lmk_cartesian = np.array(crs.transform(lmk_desc["long"], lmk_desc["lat"], 0, radians=False)[:-1])
                lmk_cartesian += translation

            landmarks[lmk] = {"x": lmk_cartesian[0], "y": lmk_cartesian[1]}

    return landmarks


def create_waypoints(obj_fpath, crs):
    """
    Create waypoints of objects similar to that created by Spot GraphNav.
    """
    objects = load_from_file(obj_fpath)

    # Set the origin of Cartesian coordinates to the location of the robot
    robot = None
    if "waypoint_0" in objects:
        robot = objects["waypoint_0"]
    else:
        print(" >> ERROR: missing robot coordinates. Check obj_fpath file")
        exit()

    if "lat" in list(objects.values())[0]:
        # Convert locations of objects from geographic to Cartesian coordinates if not provided as Cartesian
        if not crs:
            _, _, zone, _ = utm.from_latlon(robot["lat"], robot["long"])
            crs = Transformer.from_crs(crs_from="+proj=latlong +ellps=WGS84 +datum=WGS84",
                                       crs_to=f"+proj=utm +ellps=WGS84 +datum=WGS84 +south +units=m +zone={zone}")

        # NOTE: a 2D map actually is projected to the X-Z Cartesian plane, NOT X-Y
        # thus we only take the x and z coordinates, where the z will be used as Spot's y-axis
        for loc in objects.values():
            loc["x"], loc["y"] = crs.transform(loc["long"], loc["lat"], 0, radians=False)[:-1]

    # Convert to same data structure output by Spot GraphNav
    waypoints = {obj: {"name": obj, "position": {"x": loc["x"], "y": loc["y"]}} for obj, loc in objects.items()}

    return waypoints, crs


def load_lmks(graph_dpath=None, osm_fpath=None, ignore_graph=False):
    """
    Load landmarks from OSM or Spot graph or both then convert their locations to Cartesian coordinates.
    """
    # Load waypoints from Spot graph if exists
    waypoints, transformer = None, None
    try:
        graph, _, _, _, _, _ = load_map(graph_dpath)
    except Exception:
        print(" >> WARNING: no Spot graph file found in provided directory path\nCreate waypoints from object locations")
        waypoints, transformer = create_waypoints(os.path.join(graph_dpath, "obj_locs.json"), crs=None)
    else:
        # Get important details from Spot graph and create a dict instead of using their data structure
        waypoints = extract_waypoints(graph)

    # Load text description of OSM landmarks if exists
    osm_landmarks = []
    if os.path.isfile(osm_fpath):
        osm_landmarks = load_from_file(osm_fpath)

        # Use geographic coordinates of first landmark to get a zone number for UTM conversion
        lmk_desc = list(osm_landmarks.values())[0]
        _, _, zone, _ = utm.from_latlon(lmk_desc["lat"], lmk_desc["long"])
        transformer = Transformer.from_crs(crs_from="+proj=latlong +ellps=WGS84 +datum=WGS84",
                                           crs_to=f"+proj=utm +ellps=WGS84 +datum=WGS84 +south +units=m +zone={zone}")
    else:
        print(" >> WARNING: no OSM landmarks loaded")

    # Load Spot waypoints in both geographic and Cartesian coordinates for transformation
    alignment_lmks = []
    alignment_fpath = os.path.join(graph_dpath, "alignment.json")
    if os.path.isfile(alignment_fpath):
        alignment_lmks = load_from_file(alignment_fpath)

    # Put Spot waypoints and OSM landmarks in Cartesian coordinates
    landmarks = align_coordinates(graph_dpath, waypoints, osm_landmarks, alignment_lmks, transformer)

    # Visualize landmarks
    plot_landmarks(landmarks, osm_fpath)

    return landmarks


def sort_combs(lmk_grounds):
    """
    Sort all combinations of target and anchor landmarks by their joint cosine similarity scores.
    """
    combs_sorted = []

    for comb in list(product(*lmk_grounds)):  # Cartesian product of lists of target and anchor landmarks
        joint_score = 1
        target, anchor = [], []

        for idx, score_lmk in enumerate(comb):
            joint_score *= score_lmk[0]

            # Get target or anchor landmark name of the combination
            if idx == 0:  # target landmark is always the first in a combination
                target.append(score_lmk[1])
            else:  # SRE with 0, 1 or 2 target landmarks
                anchor.append(score_lmk[1])

        combs_sorted.append({"score": joint_score, "target": target, "anchor": anchor})

    combs_sorted.sort(key=lambda comb: comb["score"], reverse=True)
    return combs_sorted


def find_match_rel(rel_unseen, rel_embeds_fpath):
    """
    Use cosine similatiry between text embeddings to find best matching known spatial relation to unseen input.
    """
    if os.path.isfile(rel_embeds_fpath):
        known_rel_embeds = load_from_file(rel_embeds_fpath)
    else:
        known_rel_embeds = {known_rel: get_embed(known_rel) for known_rel in KNOWN_RELATIONS}
        save_to_file(known_rel_embeds, rel_embeds_fpath)

    unseen_rel_embed = get_embed(rel_unseen)
    scores = cosine_similarity(np.array(unseen_rel_embed).reshape(1, -1), np.array(list(known_rel_embeds.values())))[0]
    rel_match = sorted(zip(scores, KNOWN_RELATIONS), reverse=True)[0][1]
    return rel_match


def get_target_loc(landmarks, spatial_rel, anchor_candidate, sre=None, plot=False):
    """
    Ground spatial referring expression with only an anchor landmark: left, right, cardinal directions
    by finding a location relative to the given anchor landmark.
    e.g., go to the left side of the bakery, go to the north of the bakery
    """
    robot = landmarks["robot"]
    try:
        anchor = landmarks[anchor_candidate]
    except KeyError:
        return None

    # Compute valid the range vector(s) (potentially only one) for an anchoring landmark
    range_vecs = compute_area(spatial_rel, robot, anchor)

    # Compute robot location that is at given distance to the anchor
    loc_min = {"x": (range_vecs[0]["mean"][0] * DIST_TO_ANCHOR) + anchor["x"],
               "y": (range_vecs[0]["mean"][1] * DIST_TO_ANCHOR) + anchor["y"]}
    dist_min = np.linalg.norm(np.array([loc_min["x"], loc_min["y"]]) - np.array([robot["x"], robot["y"]]))
    range_vec_closest = range_vecs[0]

    for range_vec in range_vecs:
        loc_new = {"x": (range_vec["mean"][0] * DIST_TO_ANCHOR) + anchor["x"],
                   "y": (range_vec["mean"][1] * DIST_TO_ANCHOR) + anchor["y"]}
        dist_new = np.linalg.norm(np.array([loc_new["x"], loc_new["y"]]) - np.array([robot["x"], robot["y"]]))

        if dist_new < dist_min:
            loc_min = loc_new
            dist_min = dist_new
            range_vec_closest = range_vec

    if plot:
        plt.figure()

        plt.scatter(x=[robot["x"]], y=[robot["y"]], marker="o", label="robot")
        plt.scatter(x=[loc_min["x"]], y=[loc_min["y"]], marker="x", c="g", s=15, label="new robot loc")

        # Plot all target and anchor landmarks
        for A in landmarks:
            plt.scatter(x=landmarks[A]["x"], y=landmarks[A]["y"], marker="o", c="darkorange", label=f"anchor: {A}")
            plt.text(landmarks[A]["x"], landmarks[A]["y"], A)

        # Plot the range
        plt.plot([anchor["x"], (range_vec_closest["min"][0] * DIST_TO_ANCHOR) + anchor["x"]],
                 [anchor["y"], (range_vec_closest["min"][1] * DIST_TO_ANCHOR) + anchor["y"]],
                 linestyle="dotted", c="r")
        plt.plot([anchor["x"], (range_vec_closest["max"][0] * DIST_TO_ANCHOR) + anchor["x"]],
                 [anchor["y"], (range_vec_closest["max"][1] * DIST_TO_ANCHOR) + anchor["y"]],
                 linestyle="dotted", c="b")

        plt.title(f"Computed Target Position: {sre}" if sre else f"Computed Target Position: {spatial_rel}")
        plt.axis("square")
        plt.legend()
        plt.show(block=False)
    return loc_min


def compute_area(spatial_rel, robot, anchor, do_360_search=False, anchor_name=None, plot=False):
    """
    Compute a vector from anchor to robot as a normal vector pointing outside of anchor
    and a range within which the vector from anchor to target can lie.
    """
    range_vecs = []

    # Compute unit vector from anchor to robot
    vec_a2r = [robot["x"] - anchor["x"], robot["y"] - anchor["y"]]
    unit_vec_a2r = np.array(vec_a2r) / np.linalg.norm(vec_a2r)
    fov = 180  # robot's field-of-view

    if spatial_rel in ["in front of", "opposite to"]:
        mean_angle = 0
    elif spatial_rel in ["behind"]:
        mean_angle = 180
    elif spatial_rel in ["left"]:
        mean_angle = -90
    elif spatial_rel in ["right"]:
        mean_angle = 90
    elif spatial_rel in ["north of", "north", "south of", "south", "east of", "east", "west of", "west",
                         "northeast of", "northeast", "northwest of", "northwest",
                         "southeast of", "southeast", "southwest of", "southwest"]:
        # Find the difference between each cardinal direction and the current anchor-to-robot vector to figure out how much to rotate it
        if spatial_rel in ["north", "north of"]:
            mean_angle = np.rad2deg(np.arctan2(1, 0) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
        elif spatial_rel in ["south", "south of"]:
            mean_angle = np.rad2deg(np.arctan2(-1, 0) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
        elif spatial_rel in ["east", "east of"]:
            mean_angle = np.rad2deg(np.arctan2(0, 1) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
        elif spatial_rel in ["west", "west of"]:
            mean_angle = np.rad2deg(np.arctan2(0, -1) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
        elif spatial_rel in ["northeast", "northeast of"]:
            mean_angle = np.rad2deg(np.arctan2(1, 1) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
            fov = 90
        elif spatial_rel in ["northwest", "northwest of"]:
            mean_angle = np.rad2deg(np.arctan2(1, -1) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
            fov = 90
        elif spatial_rel in ["southeast", "southeast of"]:
            mean_angle = np.rad2deg(np.arctan2(-1, 1) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
            fov = 90
        elif spatial_rel in ["southwest", "southwest of"]:
            mean_angle = np.rad2deg(np.arctan2(-1, -1) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
            fov = 90

        do_360_search = False  # NOTE: since cardinal directions are absolute, we should not do any 360-sweep:

    # Check for sweep condition: this means we will consider different normal vectors representing the "front" of the object
    rots_a2r = [0]
    if spatial_rel in ["near", "next to", "adjacent to", "close to", "by"] or do_360_search:
        mean_angle = 0
        rots_a2r = [rot for rot in range(0, 360, fov)]

    for rot in rots_a2r:
        # Compute the mean vector and vectors representing min and max range
        vec_a2t_mean = rotate(unit_vec_a2r, np.deg2rad(mean_angle + rot))
        vec_a2t_min = rotate(vec_a2t_mean, np.deg2rad(- fov / 2))
        vec_a2t_max = rotate(vec_a2t_mean, np.deg2rad(fov / 2))
        range_vecs.append({"mean": vec_a2t_mean, "min": vec_a2t_min, "max": vec_a2t_max})

    if plot:
        plt.figure()

        # Plot robot and anchor location
        plt.scatter(x=[robot["x"]], y=[robot["y"]], marker="o", color="yellow", label="robot")
        plt.scatter(x=[anchor["x"]], y=[anchor["y"]], marker="o", color="orange", label="anchor")
        plt.text(anchor["x"], anchor["y"], s=anchor_name)

        # Plot the normal vector from the robot to the anchor:
        plt.plot([robot["x"], anchor["x"]], [robot["y"], anchor["y"]], color="black")
        plt.arrow(x=robot["x"], y=robot["y"], dx=-vec_a2r[0]/2.0, dy=-vec_a2r[1]/2.0, shape="full",
                  width=0.01, head_width=0.1, color="black", label="normal")

        for idx, range_vec in enumerate(range_vecs):
            mean_pose = [(range_vec["mean"][0] * MAX_RANGE) + anchor["x"],
                         (range_vec["mean"][1] * MAX_RANGE) + anchor["y"]]
            plt.scatter(x=[mean_pose[0]], y=[mean_pose[1]], c="g", marker="o", label=f"mean_{idx}")

            min_pose = [(range_vec["min"][0] * MAX_RANGE) + anchor["x"],
                        (range_vec["min"][1] * MAX_RANGE) + anchor["y"]]
            plt.scatter(x=[min_pose[0]], y=[min_pose[1]], c="r", marker="x", label=f"min_{idx}")

            max_pose = [(range_vec["max"][0] * MAX_RANGE) + anchor["x"],
                        (range_vec["max"][1] * MAX_RANGE) + anchor["y"]]
            plt.scatter(x=[max_pose[0]], y=[max_pose[1]], c="b", marker="x", label=f"max_{idx}")

            plt.plot([anchor["x"], mean_pose[0]], [anchor["y"], mean_pose[1]], linestyle="dashed", c="g")
            plt.plot([anchor["x"], min_pose[0]], [anchor["y"], min_pose[1]], linestyle="dotted", c="r")
            plt.plot([anchor["x"], max_pose[0]], [anchor["y"], max_pose[1]], linestyle="dotted", c="b")

        plt.title(f"Evaluated range for spatial relation: {spatial_rel}")
        plt.legend()
        plt.axis("square")
        plt.show(block=False)
        plt.savefig(f"compute-area_{'_'.join(spatial_rel.split(' '))}.png")
    return range_vecs


def eval_spatial_pred(landmarks, spatial_rel, target_candidate, anchor_candidates, sre=None, plot=False):
    """
    Evaluate if a spatial relation is valid given candidate target landmark and anchor landmark(s).
    """
    robot, target = landmarks["robot"], landmarks[target_candidate]

    # If target equals to any anchor, spatial predicate is True
    if target_candidate in anchor_candidates:
        return False
    # Check if target has same location as any anchor
    # OSM landmark name and Spot waypoint ID refer to same location
    for lmk_id in anchor_candidates:
        if target["x"] == landmarks[lmk_id]["x"] and target["y"] == landmarks[lmk_id]["y"]:
            return False

    if spatial_rel in ["between"]:
        try:
            anchor_1 = landmarks[anchor_candidates[0]]
            anchor_2 = landmarks[anchor_candidates[1]]
        except KeyError:
            return False  # anchor may instead be a waypoint in the Spot space

        # Avoid evaluating a target ''between'' the same anchor
        if anchor_candidates[0] == anchor_candidates[1]:
            return False
        if anchor_1["x"] == anchor_2["x"] and anchor_1["y"] == anchor_2["y"]:
            return False

        target = np.array([target["x"], target["y"]])
        anchor_1 = np.array([anchor_1["x"], anchor_1["y"]])
        anchor_2 = np.array([anchor_2["x"], anchor_2["y"]])

        # Find two lines perpendicular to the vector defined by two anchors and passing through them
        vec_a1_to_a2 = anchor_2 - anchor_1
        slope = - vec_a1_to_a2[0] / vec_a1_to_a2[1]  # line slope is negative recipical of vector slope
        # slope = np.tan([- vec_a1_to_a2[1], vec_a1_to_a2[0]])  # line slope is negative recipical of vector slope
        offset_1 = - slope * anchor_1[0] + anchor_1[1]  # -m.x + y = c
        offset_2 = - slope * anchor_2[0] + anchor_2[1]  # -m.x + y = d

        # Check target between two lines: a.x_tar + b.y_tar between c and d
        offset_tar = -slope * target[0] + target[1]

        is_tar_between = (offset_tar >= offset_1 and offset_tar <= offset_2) or (offset_tar >= offset_2 and offset_tar <= offset_1)

        # Check target within max distance to two achors
        dist_anchor1_to_tar = np.linalg.norm(target - anchor_1)
        dist_anchor2_to_tar = np.linalg.norm(target - anchor_2)

        is_pred_true = is_tar_between and dist_anchor1_to_tar <= MAX_RANGE and dist_anchor2_to_tar <= MAX_RANGE

        if is_pred_true:
            print(f'    - VALID LANDMARKS:\ttarget:{target_candidate}\tanchor:{anchor_candidates}')

            if plot:
                vec_a1_to_a2 = anchor_2 - anchor_1; vec_a1_to_a2 /= np.linalg.norm(vec_a1_to_a2)
                A, B = rotate(vec_a1_to_a2 * MAX_RANGE, np.deg2rad(-90)) + anchor_1, rotate(vec_a1_to_a2 * MAX_RANGE, np.deg2rad(90)) + anchor_1
                C, D = rotate(vec_a1_to_a2 * MAX_RANGE, np.deg2rad(-90)) + anchor_2, rotate(vec_a1_to_a2 * MAX_RANGE, np.deg2rad(90)) + anchor_2

                plt.figure(figsize=(10,6))
                plt.title(f"Grounding SRE: {sre}\n(Target:{target_candidate}, Anchors:{anchor_candidates})")

                plt.scatter(x=[target[0]], y=[target[1]], marker='o', color='green', label='target')
                plt.scatter(x=[anchor_1[0]], y=[anchor_1[1]], marker='o', color='orange', label='anchor_1')
                plt.scatter(x=[anchor_2[0]], y=[anchor_2[1]], marker='o', color='orange', label='anchor_2')

                plt.plot([A[0], anchor_1[0]], [A[1], anchor_1[1]], linestyle='dotted', c='r')
                plt.plot([C[0], anchor_2[0]], [C[1], anchor_2[1]], linestyle='dotted', c='b')
                plt.plot([B[0], anchor_1[0]], [B[1], anchor_1[1]], linestyle='dotted', c='r')
                plt.plot([D[0], anchor_2[0]], [D[1], anchor_2[1]], linestyle='dotted', c='b')
                plt.plot([anchor_1[0], anchor_2[0]], [anchor_1[1], anchor_2[1]], linestyle='dotted', c='black')

                plt.text(x=target[0], y=target[1], s=target_candidate)
                plt.text(x=anchor_1[0], y=anchor_1[1], s=anchor_candidates[0])
                plt.text(x=anchor_2[0], y=anchor_2[1], s=anchor_candidates[1])

                plt.axis('square')
                plt.show(block=False)
                plt.savefig(f"eval-spatial-pred-{spatial_rel}-{target_candidate}-{'-'.join(anchor_candidates)}.png")
        return is_pred_true
    else:
        try:
            anchor = landmarks[anchor_candidates[0]]
        except KeyError:
            return False  # anchor may instead be a waypoint in the Spot space

        is_pred_true = False
        range_vecs = compute_area(spatial_rel, robot, anchor, anchor_name=anchor_candidates[0], plot=False)
        target = np.array([target["x"], target["y"]])
        anchor = np.array([anchor["x"], anchor["y"]])

        for range_vec in range_vecs:
            vec_anc2tar = target - anchor
            vec_mean = np.array([range_vec["mean"][0], range_vec["mean"][1]])
            vec_min = np.array([range_vec["min"][0], range_vec["min"][1]])
            vec_max = np.array([range_vec["max"][0], range_vec["max"][1]])

            is_same_dir_mean = np.dot(vec_anc2tar, vec_mean) >= 0  # angle between anchor and mean vectors [-90, 90]
            is_between_min_max = np.cross(vec_min, vec_anc2tar) >= 0 and np.cross(vec_max, vec_anc2tar) <= 0  # angle between anchor and min vectors [0, 180], between anchor and max vectors [-180, 0)
            is_within_dist = np.linalg.norm(vec_anc2tar) <= MAX_RANGE

            if is_same_dir_mean and is_between_min_max and is_within_dist:
                is_pred_true = True
                break
        return is_pred_true


        is_pred_true = False
        range_vecs = compute_area(spatial_rel, robot, anchor, anchor_name=anchor_candidates[0], plot=False)

        for range_vec in range_vecs:
            min_pose = np.array([(range_vec["min"][0] * MAX_RANGE) + anchor["x"],
                                 (range_vec["min"][1] * MAX_RANGE) + anchor["y"]])
            max_pose = np.array([(range_vec["max"][0] * MAX_RANGE) + anchor["x"],
                                 (range_vec["max"][1] * MAX_RANGE) + anchor["y"]])
            mean_pose = np.array([(range_vec["mean"][0] * MAX_RANGE) + anchor["x"],
                                  (range_vec["mean"][1] * MAX_RANGE) + anchor["y"]])

            # -- checking:
            #   1) where the mean range point lies w.r.t. the spatial rel line.
            #   2) if the target position lies on the same side of the spatial rel line.
            is_within_range = False

            # Given the slope, determine which side of the line target lies
            if (max_pose[0] - min_pose[0]) != 0:
                slope = (max_pose[1] - min_pose[1]) / (max_pose[0] - min_pose[0])
            else:
                slope = (max_pose[1] - min_pose[1]) / 1.0e-25

            intercept = max_pose[1] - (slope * max_pose[0])
            computed_y_mean = (slope * mean_pose[0]) + intercept
            computed_y_target = (slope * target["x"]) + intercept

            # Source: https://math.stackexchange.com/a/324595
            if computed_y_mean > mean_pose[1] and computed_y_target > target["y"]:
                # print("below line")
                is_within_range = True
            elif computed_y_mean <= mean_pose[1] and computed_y_target <= target["y"]:
                # print("above line")
                is_within_range = True

            dist_a2t = np.linalg.norm(np.array([target["x"], target["y"]]) - np.array([anchor["x"], anchor["y"]]))

            if is_within_range and dist_a2t <= MAX_RANGE:
                print(f"    - VALID LANDMARKS:\ttarget:{target_candidate}\tanchor:{anchor_candidates[0]}")
                is_pred_true = True
                break

        if is_pred_true:
            if plot:
                # Plot the computed vector range
                plt.figure(figsize=(10,6))
                plt.title(f"Grounding SRE: {sre}\n(Target:{target_candidate}, Anchor:{anchor_candidates})")

                plt.scatter(x=[robot["x"]], y=[robot["y"]], marker="o", color="yellow", label="robot")
                plt.scatter(x=[anchor["x"]], y=[anchor["y"]], marker="o", color="orange", label="anchor")
                plt.scatter(x=[target["x"]], y=[target["y"]], marker="o", color="green", label="target")

                plt.plot([robot["x"], anchor["x"]], [robot["y"], anchor["y"]], linestyle="dotted", c="k", label="normal")

                plt.text(anchor["x"], anchor["y"], s=anchor_candidates[0])
                plt.text(target["x"], target["y"], s=target_candidate)

                for vec_idx, range_vec in enumerate(range_vecs):
                    mean_pose = np.array([(range_vec["mean"][0] * MAX_RANGE) + anchor["x"],
                                          (range_vec["mean"][1] * MAX_RANGE) + anchor["y"]])
                    plt.scatter(x=[mean_pose[0]], y=[mean_pose[1]], c="grey", marker="x", label="mean")

                    min_pose = np.array([(range_vec["min"][0] * MAX_RANGE) + anchor["x"],
                                         (range_vec["min"][1] * MAX_RANGE) + anchor["y"]])
                    plt.scatter(x=[min_pose[0]], y=[min_pose[1]], c="r", marker="x", label="min")

                    max_pose = np.array([(range_vec["max"][0] * MAX_RANGE) + anchor["x"],
                                         (range_vec["max"][1] * MAX_RANGE) + anchor["y"]])
                    plt.scatter(x=[max_pose[0]], y=[max_pose[1]], c="b", marker="x", label="max")

                    if vec_idx == (len(range_vecs) - 1):
                        plt.plot([anchor["x"], mean_pose[0]], [anchor["y"], mean_pose[1]], linestyle="dotted", c="grey", label="mean_range" )
                        plt.plot([anchor["x"], min_pose[0]], [anchor["y"], min_pose[1]], linestyle="dotted", c="r", label="min_range")
                        plt.plot([anchor["x"], max_pose[0]], [anchor["y"], max_pose[1]], linestyle="dotted", c="b", label="max_range")
                    else:
                        plt.plot([anchor["x"], mean_pose[0]], [anchor["y"], mean_pose[1]], linestyle="dotted", c="grey")
                        plt.plot([anchor["x"], min_pose[0]], [anchor["y"], min_pose[1]], linestyle="dotted", c="r")
                        plt.plot([anchor["x"], max_pose[0]], [anchor["y"], max_pose[1]], linestyle="dotted", c="b")

                plt.legend()
                plt.axis("square")
                plt.show(block=False)
            return True
    return False


def spg(landmarks, reg_out, topk, rel_embeds_fpath, max_range=None):
    print(f"***** SPG Command: {reg_out['utt']}")

    if max_range:
        global MAX_RANGE
        MAX_RANGE = max_range
    print(f" -> MAX_RANGE = {MAX_RANGE}\n")

    spg_out = {}

    for sre, grounded_spatial_preds in reg_out["grounded_sre_to_preds"].items():
        print(f"Grounding SRE: {sre}")

        rel_query, lmk_grounds = list(grounded_spatial_preds.items())[0]

        # Rank all combinations of target and anchor landmarks
        lmk_grounds_sorted = sort_combs(lmk_grounds)

        if rel_query == "None":
            # Referring expression without spatial relation
            groundings = [{"target": lmk_ground["target"][0]} for lmk_ground in lmk_grounds_sorted[:topk]]
        else:
            groundings = []

            rel_match = rel_query
            if rel_query not in KNOWN_RELATIONS:
                # Find best match for unseen spatial relation in set of known spatial relations
                rel_match = find_match_rel(rel_query, rel_embeds_fpath)
                print(f"UNSEEN SPATIAL RELATION:\t'{rel_query}' matched to '{rel_match}'")

            if len(lmk_grounds) == 1:
                # Spatial referring expression contains only a anchor landmark
                for lmk_ground in lmk_grounds_sorted[:topk]:
                    groundings.append(get_target_loc(landmarks, rel_match, lmk_ground["target"][0], sre))
            else:
                # Spatial referring expression contains a target landmark and one or two anchor landmarks
                # one anchor, e.g., <tar> left of <anc>
                # two anchors, e.g., <tar> between <anc1> and <anc2>
                for lmk_ground in lmk_grounds_sorted:
                    target_name = lmk_ground["target"][0]
                    anchor_names = lmk_ground["anchor"]
                    is_valid = eval_spatial_pred(landmarks, rel_match, target_name, anchor_names, sre)
                    if is_valid:
                        groundings.append({"target": target_name,  "anchor": anchor_names})
                    if len(groundings) == topk:
                        break

        spg_out[sre] = groundings

        plt.close("all")
        print("\n")
    return spg_out


def run_exp_spg(reg_out_fpath, graph_dpath, osm_fpath, topk, rel_embeds_fpath, spg_out_fpath):
    if not os.path.isfile(spg_out_fpath):
        reg_outs = load_from_file(reg_out_fpath)
        landmarks = load_lmks(graph_dpath, osm_fpath)
        for reg_out in tqdm(reg_outs, desc="Running spatial predicate grounding (SPG) module"):
            reg_out["grounded_sps"] = spg(landmarks, reg_out, topk, rel_embeds_fpath)
        save_to_file(reg_outs, spg_out_fpath)


if __name__ == "__main__":
    location = "blackstone"
    data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
    graph_dpath = os.path.join(data_dpath, "maps", "downloaded_graph_2024-01-27_07-48-53")
    osm_fpath = os.path.join(data_dpath, "osm", f"{location}.json")
    reg_outs_fpath = os.path.join(os.path.expanduser("~"), "ground", "results", f"reg_outs_{location}.json")

    reg_outputs = load_from_file(reg_outs_fpath)
    landmarks = load_lmks(graph_dpath, osm_fpath)
    for reg_output in reg_outputs:
        spg(landmarks, reg_output, topk=5)
