import os
import numpy as np
import matplotlib.pyplot as plt
import itertools as it

from load_map import load_map
from openai_models import get_embed
from utils import load_from_file


# NOTE: spot_waypoints_path :- path to the folder of files generated by GraphNav:
# spot_graph_path = os.path.join(os.path.expanduser("~"), "ground", "spot", "graphs", "blackstone")

# NOTE: osm_path :- file path for the JSON file containing OSM-extracted landmarks:
osm_path = os.path.join(os.path.expanduser("~"), "ground", "data", "osm", "blackstone.json")

# NOTE: reg_output_path :- file path to the output of the RER process (obtained from Jason):
reg_output_path = os.path.join(os.path.expanduser("~"), "ground", "data", "reg_outs_blackstone.json")

known_spatial_relations = [
    'left', 'left of', 'to the left of', 'right', 'right of', 'to the right of',
    'in front of', 'opposite', 'opposite to', 'behind', 'behind of', 'at the rear of',
    'near', 'near to', 'next', 'next to', 'adjacent to', 'close', 'close to', 'at', 'by', 'between',
    'north of', 'south of', 'east of', 'west of', 'northeast of', 'northwest of', 'southeast of', 'southwest of'
]

grounding_landmark = [
    {
        'waypoint': 'rose-barker-SI61poibGHSEpOdZa0lzmQ==',
        'gps': {'lat': 41.858055308939804, 'long':-71.39119853826014}
    },
    {
        # NOTE: chair: https://maps.app.goo.gl/FAUaqQRXQar9J1P79
        'waypoint' : 'lossy-beef-zbwSxEF1a4R7bkFu3V9Cyg==',
        'gps': {'lat': 41.85814693862552, 'long':-71.39127870267093}
    },
    {
        # NOTE: bicycle rack at Blackstone Plaza: https://maps.app.goo.gl/m9iH3Mzj5PVrjeJq9
        'waypoint': 'pyknic-auk-eHDSWo4QNapDsHViTQRztw==',              # -- this is the name of the Spot waypoint for which we know its GPS position
        'gps': {'lat': 41.85809499132926, 'long': -71.39110567075839}   # -- this is the GPS location for the waypoint above
    },
]

# NOTE: robot will always start at what was marked "waypoint_0":
robot = None

landmarks = None

use_pyproj = False

# -- let's assume that we will only look for an object that is within 10m of the anchor:
max_range = 50.0
range_to_anchor = 2.0

def rotation_matrix(angle):
    # Source: https://motion.cs.illinois.edu/RoboticSystems/CoordinateTransformations.html
    return np.array(
        [[np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]])
#enddef


def gps_to_cartesian(landmark):
    # Source: https://stackoverflow.com/questions/1185408/converting-from-longitude-latitude-to-cartesian-coordinates

    lat, long = landmark['lat'], landmark['long']

    # NOTE: radius of earth is approximately 6371 km; if we want it scaled to meters, then we should multiply by 1000:
    radius_earth = 6378.1370 * 1000
    x = radius_earth * np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(long))
    y = radius_earth * np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(long))
    z = radius_earth * np.sin(np.deg2rad(lat))
    return [x, y, z]
#enddef


def find_closest_relation(rel):
    # -- using text embedding model provided by OpenAI:
    closest_rel = None
    closest_rel_embedding = None

    # -- precompute the embedding for the unseen relation:
    unseen_rel_embedding = get_embed(rel)

    for R in known_spatial_relations:

        # -- get an embedding for each predefined relation:
        candidate_embedding = get_embed(R)

        if not closest_rel:
            closest_rel = R
            closest_rel_embedding = candidate_embedding

        else:
            # -- compute cosine similarity between words and pick the higher (i.e., most similar) word:
            current_score = np.dot(unseen_rel_embedding, closest_rel_embedding)
            new_rel_score = np.dot(unseen_rel_embedding, candidate_embedding)

            if current_score < new_rel_score:
                closest_rel = R
                closest_rel_embedding =candidate_embedding

    return closest_rel
#enddef


def compute_area(spatial_rel, anchor, do_360_search=False, plot=False):
    list_ranges = []

    # NOTE: we want to draw a vector from the anchor's perspective to the robot!
    # -- this gives us a normal vector pointing outside of the anchor object

    # -- compute vector between robot's position and anchor position and get its direction:
    vector_a2r = [robot['x'] - anchor['x'],
                    robot['y'] - anchor['y']]

    # -- draw a unit vector and multiply it by 10 to get the max distance to consider:
    unit_vec_a2r = np.array(vector_a2r) / np.linalg.norm(vector_a2r)

    # NOTE: mean angle of 0 if we get the spatial relation "in front of" or "opposite"
    mean_angle = 0
    if spatial_rel in ['left', 'left of', 'to the left of']:
        # -- if we want something to the left, we need to go in positive 90 degrees:
        mean_angle = -90
    elif spatial_rel in ['right', 'right of', 'to the right of']:
        # -- if we want something to the right, we need to go in negative 90 degrees:
        mean_angle = 90
    elif spatial_rel in ['behind', 'at the rear of', 'behind of']:
        # -- if we want something to the right, we need to tn 180 degees:
        mean_angle = 180
    elif spatial_rel in ['north of', 'south of', 'east of', 'west of', 'northeast of', 'northwest of', 'southeast of', 'southwest of']:
        # -- we need to find the difference between each cardinal direction and the current anchor-to-robot vector
        #       to figure out how much we need to rotate it by:
        if spatial_rel in ['north', 'north of']:
            mean_angle = np.rad2deg(np.arctan2(1, 0) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
        elif spatial_rel in ['south', 'south of']:
            mean_angle = np.rad2deg(np.arctan2(-1, 0) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
        elif spatial_rel in ['east', 'east of']:
            mean_angle = np.rad2deg(np.arctan2(0, 1) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
        elif spatial_rel in ['west', 'west of']:
            mean_angle = np.rad2deg(np.arctan2(0, -1) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
        elif spatial_rel in ['northeast', 'northeast of']:
            mean_angle = np.rad2deg(np.arctan2(1, 1) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
        elif spatial_rel in ['northwest', 'northwest of']:
            mean_angle = np.rad2deg(np.arctan2(1, -1) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
        elif spatial_rel in ['southeast', 'southeast of']:
            mean_angle = np.rad2deg(np.arctan2(-1, 1) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
        elif spatial_rel in ['southwest', 'southwest of']:
            mean_angle = np.rad2deg(np.arctan2(-1, -1) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))

        # NOTE: since cardinal directions are absolute, we should not do any 360-sweep:
        do_360_search = False

    # endif

    # -- this dictates how wide of a field-of-view we attribute to the robot:
    field_of_view = 180

    # -- checking for sweep condition: this means we will consider different normal vectors
    #       representing the "front" of the object:
    rot_a2r = [0]
    if spatial_rel in ['near', 'near to', 'next', 'next to', 'adjacent to', 'close to', 'at', 'close', 'by'] or do_360_search:
        rot_a2r += [x * field_of_view for x in range(1, int(360 / field_of_view))]

    # print(rot_a2r)
    for x in rot_a2r:
        # -- rotate the anchor's frame of reference by some angle x:
        a2r_vector = np.dot(rotation_matrix(
            angle=np.deg2rad(x)), unit_vec_a2r)

        # -- compute the mean vector as well as vectors representing min and max proximity range:
        a2r_mean = np.dot(rotation_matrix(
            angle=np.deg2rad(mean_angle)), a2r_vector)
        a2r_min_range = np.dot(rotation_matrix(
            angle=np.deg2rad(mean_angle-(field_of_view/2))), a2r_vector)
        a2r_max_range = np.dot(rotation_matrix(
            angle=np.deg2rad(mean_angle+(field_of_view/2))), a2r_vector)

        # -- append the vectors to the list of evaluated ranges:
        list_ranges.append({
            'mean': a2r_mean,
            'min': a2r_min_range,
            'max': a2r_max_range,
        })
    # endfor

    if plot:
        # -- plot the computed range:
        plt.figure()
        plt.title(f'Evaluated range for spatial relation "{spatial_rel}"')
        plt.scatter(x=[robot['x']], y=[robot['y']], marker='o', color='yellow', label='robot')
        plt.scatter(x=[anchor['x']], y=[anchor['y']], marker='o', color='orange', label='anchor')
        plt.text(anchor['x'], anchor['y'], s=anchor['name'])

        plt.plot([robot['x'], anchor['x']], [
                    robot['y'], anchor['y']], color='black')
        plt.arrow(x=robot['x'], y=robot['y'], dx=-vector_a2r[0]/2.0, dy=-vector_a2r[1]/2.0, shape='full',
                    width=0.01, head_width=0.1, color='black', label='normal')

        for r in range(len(list_ranges)):
            mean_pose = [(list_ranges[r]['mean'][0] * max_range) + anchor['x'],
                            (list_ranges[r]['mean'][1] * max_range) + anchor['y']]
            plt.scatter(x=[mean_pose[0]], y=[mean_pose[1]],
                        c='g', marker='o', label=f'mean_{r}')

            min_pose = [(list_ranges[r]['min'][0] * max_range) + anchor['x'],
                        (list_ranges[r]['min'][1] * max_range) + anchor['y']]
            plt.scatter(x=[min_pose[0]], y=[min_pose[1]],
                        c='r', marker='x', label=f'min_{r}')

            max_pose = [(list_ranges[r]['max'][0] * max_range) + anchor['x'],
                        (list_ranges[r]['max'][1] * max_range) + anchor['y']]
            plt.scatter(x=[max_pose[0]], y=[max_pose[1]],
                        c='b', marker='x', label=f'max_{r}')

            plt.plot([anchor['x'], mean_pose[0]], [
                        anchor['y'], mean_pose[1]], linestyle='dashed', c='g')
            plt.plot([anchor['x'], min_pose[0]], [
                        anchor['y'], min_pose[1]], linestyle='dotted', c='r')
            plt.plot([anchor['x'], max_pose[0]], [
                        anchor['y'], max_pose[1]], linestyle='dotted', c='b')
        # endfor

        plt.legend()
        plt.axis('square')
        plt.show(block=False)

    return list_ranges
# enddef


def evaluate_spg(spatial_rel, target_candidate, anchor_candidates, sre=None, plot=False, do_360_search=False):

    global robot, landmarks, max_range

    # -- we cannot evaluate a landmark against itself:
    if target_candidate in anchor_candidates:
        return False

    # -- in this case, we will be given a list of target objects or entities:
    target = landmarks[target_candidate]

    if spatial_rel not in ['between']:

        try:
            anchor = landmarks[anchor_candidates[0]]
        except KeyError:
            return False

        anchor['name'] = anchor_candidates[0]

        list_ranges = compute_area(spatial_rel, anchor, plot=plot)

        is_valid = False

        for R in list_ranges:
            dir_tgt = np.rad2deg(np.arctan2(target['y'] - anchor['y'], target['x'] - anchor['x']))
            dir_min = np.rad2deg(np.arctan2(R['min'][1], R['min'][0]))
            dir_max = np.rad2deg(np.arctan2(R['max'][1], R['max'][0]))

            # -- making it easier by making things w.r.t. 0 - 360 degree measurements:
            if dir_tgt < 0:
                dir_tgt += 360
            if dir_max < 0:
                dir_max += 360
            if dir_min < 0:
                dir_min += 360

            # -- this is a case where the max vector is positive while the min vector is negative, resulting in it being larger:
            if dir_min > dir_max:
                dir_max += 360

            distance_a2t = np.linalg.norm(np.array([target['x'], target['y']]) - np.array([anchor['x'], anchor['y']]))

            if dir_tgt >= dir_min and dir_tgt <= dir_max and distance_a2t < max_range:
                print(f'    - VALID LANDMARKS:\ttarget:{target_candidate}\tanchor:{anchor_candidates[0]}')
                is_valid = True
                break

        if is_valid:

            if plot:
                # -- plot the computed range:
                plt.figure()
                plt.title(f'Final Grounding: "{sre}"\n(Target:{target_candidate}, Anchor:{anchor_candidates})')
                plt.scatter(x=[robot['x']], y=[robot['y']], marker='o', color='yellow', label='robot')
                plt.scatter(x=[anchor['x']], y=[anchor['y']], marker='o', color='orange', label='anchor')
                plt.scatter(x=[target['x']], y=[target['y']], marker='o', color='green', label='target')
                plt.plot([robot['x'], anchor['x']], [robot['y'], anchor['y']], linestyle='dotted', c='k', label='normal')

                plt.text(anchor['x'], anchor['y'], s=anchor_candidates[0])
                plt.text(target['x'], target['y'], s=target_candidate)

                for R in range(len(list_ranges)):
                    mean_pose = np.array([(list_ranges[R]['mean'][0] * max_range) + anchor['x'],
                                    (list_ranges[R]['mean'][1] * max_range) + anchor['y']])
                    # plt.scatter(x=[mean_pose[0]], y=[mean_pose[1]], c='g', marker='o', label='mean')

                    min_pose = np.array([(list_ranges[R]['min'][0] * max_range) + anchor['x'],
                                (list_ranges[R]['min'][1] * max_range) + anchor['y']])
                    # plt.scatter(x=[min_pose[0]], y=[min_pose[1]], c='r', marker='x', label='min')

                    max_pose = np.array([(list_ranges[R]['max'][0] * max_range) + anchor['x'],
                                (list_ranges[R]['max'][1] * max_range) + anchor['y']])
                    # plt.scatter(x=[max_pose[0]], y=[max_pose[1]], c='b', marker='x', label='max')

                    if R == (len(list_ranges) - 1):
                        # plt.plot([anchor['x'], mean_pose[0]], [anchor['y'], mean_pose[1]], linestyle='dotted', c='g', label='mean_range' )
                        plt.plot([anchor['x'], min_pose[0]], [anchor['y'], min_pose[1]], linestyle='dotted', c='r', label='min_range')
                        plt.plot([anchor['x'], max_pose[0]], [anchor['y'], max_pose[1]], linestyle='dotted', c='b', label='max_range')
                    else:
                        # plt.plot([anchor['x'], mean_pose[0]], [anchor['y'], mean_pose[1]], linestyle='dotted', c='g', )
                        plt.plot([anchor['x'], min_pose[0]], [anchor['y'], min_pose[1]], linestyle='dotted', c='r')
                        plt.plot([anchor['x'], max_pose[0]], [anchor['y'], max_pose[1]], linestyle='dotted', c='b')

                plt.legend()
                plt.axis('square')
                plt.show(block=False)

            return True

    else:

        try:
            anchor_1 = landmarks[anchor_candidates[0]]
            anchor_2 = landmarks[anchor_candidates[1]]
        except KeyError:
            # -- this anchor may instead be a waypoint in the Spot's space:
            return False

        if anchor_candidates[0] == anchor_candidates[1] or anchor_1 == anchor_2:
            return False


        # -- checking if something lies between two anchors is fairly simple: https://math.stackexchange.com/a/190373

        target = np.array([target['x'], target['y']])
        anchor_1, anchor_2 = np.array([anchor_1['x'], anchor_1['y']]), np.array([anchor_2['x'], anchor_2['y']])

        # -- computing vectors perpendicular to each anchoring point:
        vec_a1_to_a2 = anchor_2 - anchor_1; vec_a1_to_a2 /= np.linalg.norm(vec_a1_to_a2)
        vec_a2_to_a1 = anchor_1 - anchor_2; vec_a2_to_a1 /= np.linalg.norm(vec_a2_to_a1)
        A, B = np.dot(rotation_matrix(np.deg2rad(-90)), vec_a1_to_a2 * max_range) + anchor_1, np.dot(rotation_matrix(np.deg2rad(90)), vec_a1_to_a2 * max_range) + anchor_1
        C, D = np.dot(rotation_matrix(np.deg2rad(-90)), vec_a2_to_a1 * max_range) + anchor_2, np.dot(rotation_matrix(np.deg2rad(90)), vec_a2_to_a1 * max_range) + anchor_2

        dot_ABAM = np.dot(B-A, target-A)
        dot_ABAB = np.dot(B-A, B-A)
        dot_BCBM = np.dot(C-B, target-B)
        dot_BCBC = np.dot(C-B, C-B)

        if 0 <= dot_ABAM and dot_ABAM <= dot_ABAB and 0 <= dot_BCBM and dot_BCBM <= dot_BCBC:
            if plot:
                plt.figure()

                plt.scatter(x=[robot['x']], y=[robot['y']], marker='o', color='yellow', label='robot')
                plt.scatter(x=[target[0]], y=[target[1]], marker='o', color='green', label='target')
                plt.scatter(x=[anchor_1[0]], y=[anchor_1[1]], marker='o', color='orange', label='anchor')
                plt.scatter(x=[anchor_2[0]], y=[anchor_2[1]], marker='o', color='orange', label='anchor')

                plt.plot([A[0], anchor_1[0]], [A[1], anchor_1[1]], linestyle='dotted', c='r')
                plt.plot([C[0], anchor_2[0]], [C[1], anchor_2[1]], linestyle='dotted', c='b')
                plt.plot([B[0], anchor_1[0]], [B[1], anchor_1[1]], linestyle='dotted', c='r')
                plt.plot([D[0], anchor_2[0]], [D[1], anchor_2[1]], linestyle='dotted', c='b')

                plt.text(x=target[0], y=target[1], s=target_candidate)
                plt.text(x=anchor_1[0], y=anchor_1[1], s=anchor_candidates[0])
                plt.text(x=anchor_2[0], y=anchor_2[1], s=anchor_candidates[1])

                plt.title(f'Final Grounding: "{sre}"\n(Target:{target_candidate}, Anchor:{anchor_candidates})')
                plt.axis('square')
                plt.show(block=False)

            print(f'    - VALID LANDMARKS:\ttarget:{target_candidate}\tanchor:{anchor_candidates}')
            return True

        return False

    # endif
    return False
# enddef


def get_target_position(spatial_rel, anchor_candidate, sre=None, plot=False):
    # -- this means that we have no target landmark: we solely want to find a position relative to a given anchor
    try:
        anchor = landmarks[anchor_candidate]
    except KeyError:
        return None

    # -- get the list of valid ranges (potentially only one) for an anchoring landmark:
    list_ranges = compute_area(spatial_rel, anchor)

    # -- we want to find the closest point from the robot to the anchoring landmark that satisfies the given spatial relation:
    closest_position = 0
    for R in range(len(list_ranges)):

        cur_min_pos = {'x': (list_ranges[R]['mean'][0] * range_to_anchor) + anchor['x'], 'y': (
            list_ranges[R]['mean'][1] * range_to_anchor) + anchor['y']}
        cur_min_dist = np.linalg.norm(np.array([cur_min_pos['x'], cur_min_pos['y']]) - np.array([robot['x'], robot['y']]))

        new_min_pos = {'x': (list_ranges[closest_position]['mean'][0] * range_to_anchor) + anchor['x'], 'y': (
            list_ranges[closest_position]['mean'][1] * range_to_anchor) + anchor['y']}
        new_min_dist = np.linalg.norm(np.array([new_min_pos['x'], new_min_pos['y']]) - np.array([robot['x'], robot['y']]))

        if cur_min_dist > new_min_dist:
            new_min_pos = R
    # endfor

    # -- select the index that was found to be closest to the robot:
    R = list_ranges[closest_position]

    # -- use the mean vector to find a point that is within range_to_anchor (2m) of the anchor:
    new_robot_pos = {'x': (R['mean'][0] * range_to_anchor) + anchor['x'],
                        'y': (R['mean'][1] * range_to_anchor) + anchor['y']}

    if plot:
        plt.figure()
        plt.title(f'Computed Target Position: "{sre}"' if sre else f'Computed Target Position: "{spatial_rel}"')
        plt.scatter(x=[robot['x']], y=[robot['y']], marker='o', label='robot')
        plt.scatter(x=[new_robot_pos['x']], y=[new_robot_pos['y']], marker='x', c='g', s=15, label='new robot pose')

        # -- plot all anchors and targets provided to the function:
        for A in landmarks:
            plt.scatter(x=landmarks[A]['x'], y=landmarks[A]['y'],
                        marker='o', c='darkorange',
                        label=f"anchor: {A}")
            plt.text(landmarks[A]['x'], landmarks[A]['y'], A)

        # -- plot the range as well for visualization:
        plt.plot([anchor['x'], (R['min'][0] * range_to_anchor) + anchor['x']], [anchor['y'],
                    (R['min'][1] * range_to_anchor) + anchor['y']], linestyle='dotted', c='r')
        plt.plot([anchor['x'], (R['max'][0] * range_to_anchor) + anchor['x']], [anchor['y'],
                    (R['max'][1] * range_to_anchor) + anchor['y']], linestyle='dotted', c='b')

        plt.axis('square')
        plt.legend()
        plt.show(block=False)

    return new_robot_pos
#enddef


def plot_landmarks(landmarks=None):
    # -- plotting the points in the shared world space local to the Spot's map:
    plt.figure()

    if landmarks:
        plt.scatter(x=[landmarks[L]['x'] for L in landmarks], y=[landmarks[L]['y'] for L in landmarks], c='green', label='landmarks')
        for L in landmarks:
            if 'osm_name' not in landmarks[L] and L != 'waypoint_0':
                plt.text(landmarks[L]['x'], landmarks[L]['y'], L)

    if robot:
        plt.scatter(x=robot['x'], y=robot['y'], c='orange', label='robot')
        plt.text(robot['x'], robot['y'], 'robot')

    plt.title('Landmarks: Target and Anchor')
    plt.legend()
    plt.axis('square')
    plt.show(block=False)
#enddef


def align_coordinates(spot_graph_dpath, osm_landmarks, spot_waypoints, grounding_landmark):

    global use_pyproj, robot

    use_pyproj = False

    # NOTE: a 2D map actually is projected to the X-Z Cartesian plane, NOT X-Y:
    # -- for this reason, we only take the x and z coordinates, where the z will be used as Spot's y-axis:
    if use_pyproj:
        # -- using pyproj: https://stackoverflow.com/a/69604627
        from pyproj import Transformer
        llh_to_xyz = Transformer.from_crs("+proj=latlong +ellps=WGS84 +datum=WGS84", "+proj=geocent +ellps=WGS84 +datum=WGS84")

        known_landmark_1 = np.array(llh_to_xyz.transform(grounding_landmark[0]['gps']['long'], grounding_landmark[0]['gps']['lat'], 30, radians=False)[:-1])
        known_landmark_2 = np.array(llh_to_xyz.transform(grounding_landmark[1]['gps']['long'], grounding_landmark[1]['gps']['lat'], 30, radians=False)[:-1])
    else:
        known_landmark_1 = np.array([gps_to_cartesian(grounding_landmark[0]['gps'])[x] for x in [0, 2]])
        known_landmark_2 = np.array([gps_to_cartesian(grounding_landmark[1]['gps'])[x] for x in [0, 2]])

    known_waypoint_1 = np.array([spot_waypoints[grounding_landmark[0]['waypoint']].waypoint_tform_ko.position.x,
                              spot_waypoints[grounding_landmark[0]['waypoint']].waypoint_tform_ko.position.y])
    known_waypoint_2 = np.array([spot_waypoints[grounding_landmark[1]['waypoint']].waypoint_tform_ko.position.x,
                              spot_waypoints[grounding_landmark[1]['waypoint']].waypoint_tform_ko.position.y])

    # -- use the vector from known landmarks to determine the degree of rotation needed:
    vec_lrk_1_to_2 = known_landmark_2 - known_landmark_1
    vec_way_1_to_2 = known_waypoint_2 - known_waypoint_1

    # -- first, we will find the rotation between the known landmark and waypoint:
    dir_robot = np.arctan2(vec_way_1_to_2[1], vec_way_1_to_2[0])    # i.e., spot coordinate
    dir_world = np.arctan2(vec_lrk_1_to_2[1], vec_lrk_1_to_2[0])    # i.e., world coordinate

    angle_diff = dir_world - dir_robot

    # -- use the name of the images taken at each waypoint to indicate the actual waypoints of interest:
    global spot_waypoints_path

    # -- each image is named after the Spot waypoint name (auto-generated by GraphNav):
    list_waypoints = [os.path.splitext(os.path.basename(W))[0] for W in os.listdir(os.path.join(spot_graph_dpath, 'images'))]

    landmarks = {}

    for W in spot_waypoints:
        if W in list_waypoints or spot_waypoints[W].annotations.name == 'waypoint_0':
            # -- just get the x-coordinate and y-coordinate, which would correspond to a top-view 2D map:
            spot_coordinate = np.array([spot_waypoints[W].waypoint_tform_ko.position.x,
                                    spot_waypoints[W].waypoint_tform_ko.position.y])

            # -- align the Spot's coordinates to the world frame:
            spot_coordinate = np.dot(rotation_matrix(angle=angle_diff), spot_coordinate)

            # -- we will use the newly rotated point to figure out the offset:
            if W == grounding_landmark[0]['waypoint']:
                known_waypoint_1 = spot_coordinate
            elif W == grounding_landmark[1]['waypoint']:
                known_waypoint_2 = spot_coordinate

            if spot_waypoints[W].annotations.name == 'waypoint_0':
                id_name = 'waypoint_0'
                robot = {
                    'x': spot_coordinate[0],
                    'y': spot_coordinate[1],
                }
            else:
                id_name = W

            landmarks[id_name] = {
                'x': spot_coordinate[0],
                'y': spot_coordinate[1],
            }

    # -- compute an offset that can be used to align the known landmark from world to Spot space:
    offset = ((known_waypoint_1 - known_landmark_1) + (known_waypoint_2 - known_landmark_2)) / 2.0

    for L in osm_landmarks:
        # -- replace whitespace with underscore, make lowercase:
        id_name = str(L).lower().replace(' ', '_')

        if 'wid' not in osm_landmarks[L]:
            # -- we first need to convert each point into its Cartesian equivalent, then add the computed offset from above:
            if use_pyproj:
                landmark_cartesian = np.array(llh_to_xyz.transform(osm_landmarks[L]['long'], osm_landmarks[L]['lat'], 30, radians=False)[:-1])
            else:
                landmark_cartesian = np.array([gps_to_cartesian(osm_landmarks[L])[x] for x in [0, 2]])

            landmark_cartesian += offset

        else:
            # NOTE: some points in OSM will have a waypoint associated with it, so just use that x and y:
            landmark_cartesian = np.array([landmarks[osm_landmarks[L]['wid']]['x'], landmarks[osm_landmarks[L]['wid']]['y']])
            landmarks[osm_landmarks[L]['wid']]['osm_name'] = id_name

        landmarks[id_name] = {
            'x': landmark_cartesian[0],
            'y': landmark_cartesian[1],
        }

    return landmarks
#enddef


def init(spot_graph_dpath, osm_landmark_file=None):
    global landmarks

    # -- load the waypoints recorded using Spot's GraphNav:
    (_, waypoints, _, _, _, _) = load_map(spot_graph_dpath)

    if osm_landmark_file:
        osm_path = osm_landmark_file

    # -- load the OSM landmarks from a JSON file:
    osm_landmarks = load_from_file(osm_path)

    # -- iterate through all of the Spot waypoints as well as the OSM landmarks and put them in the same space:
    landmarks = align_coordinates(spot_graph_dpath, osm_landmarks, waypoints, grounding_landmark=grounding_landmark)
#enddef


def sort_by_scores(spatial_pred_dict):
    # -- find Cartesian product to find all combinations of target and anchoring landmarks:
    all_products = [x for x in it.product(*spatial_pred_dict)]

    sorted_products = []
    for P in all_products:
        # -- compute the score attributed to a given set:
        joint_score = 1
        target, anchor = [], []
        for x in range(len(P)):
            joint_score *= P[x][0]

            # -- deriving the names of the landmarks from the dictionary:
            if x == 0:
                target.append(P[x][1])
            else:
                anchor.append(P[x][1])

        # -- save the score to a list:
        sorted_products.append({'score': joint_score, 'target': target, 'anchor': anchor})

    # -- sort everything in descending order:
    sorted_products.sort(key=lambda x: x['score'], reverse=True)

    return sorted_products
#enddef


def spg(spatial_preds, topk=5):

    global landmarks, known_spatial_relations

    # -- plot the points for visualization purposes:
    plot_landmarks(landmarks)

    # -- find the closest waypoint to each anchor from OSM:
    # anchor_to_target = {}
    # for A in anchor_landmarks:
    #     this_anchor = np.array([anchor_landmarks[A]['x'], anchor_landmarks[A]['y']])

    #     closest_waypoint = None
    #     for target in target_landmarks:
    #         if not closest_waypoint:
    #             closest_waypoint = target
    #             best_waypoint = np.array([target_landmarks[closest_waypoint]['x'], target_landmarks[closest_waypoint]['y']])
    #         else:
    #             # -- check if the current target (waypoint) is closer than the closest target stored as a variable:
    #             this_waypoint = np.array([target_landmarks[target]['x'], target_landmarks[target]['y']])
    #             best_waypoint = np.array([target_landmarks[closest_waypoint]['x'], target_landmarks[closest_waypoint]['y']])

    #             # NOTE: closest in terms of Euclidean distance:
    #             if np.linalg.norm(this_anchor-this_waypoint) <= np.linalg.norm(this_anchor-best_waypoint):
    #                 closest_waypoint = target

    #     # NOTE: the best target will be the waypoint closest to the anchor:
    #     anchor_to_target[A] = closest_waypoint
    #     # print(A, anchor_to_target[A])

    spg_output = []

    for R in spatial_preds['grounded_sre_to_preds']:
        # -- extract the name of the SRE, which is a key in spatial_preds['grounded_sre_to_preds']:
        sre = R
        print(f' >> {sre}')

        # NOTE: the spatial relation is always the first key:
        unmatched_rel = list(spatial_preds['grounded_sre_to_preds'][R].keys()).pop()

        # -- extract the name of the relation in the SRE dict:
        reg_dict = spatial_preds['grounded_sre_to_preds'][R][unmatched_rel]

        # -- rank all sets of targets and anchors for evaluating spatial predicate grounding:
        grounding_set = sort_by_scores(reg_dict)

        if unmatched_rel == 'None':
            # TODO: what to do for non-spatial referring expressions?
            output = {
                'sre' : sre,
                'groundings': []
            }
            for G in grounding_set:
                output['groundings'].append({
                        'target': G['target']
                    })

            spg_output.append(output)
            continue

        # -- relation is the string that *should* be in the listed of evaluated predicates:
        relation = unmatched_rel

        # TODO: check if spatial relation is predefined:
        if unmatched_rel not in known_spatial_relations:
            # -- find the closest spatial relation:
            relation = find_closest_relation(unmatched_rel)
            print(f'    - UNSEEN RELATION:\t"{unmatched_rel}" is closest to "{relation}"!')

        if len(reg_dict) == 1:
            # NOTE: this means we only have an anchoring landmark and no target landmark:

            # -- we will keep a list of target positions in order of confidence scores from REG:
            output = {
                'sre': sre,
                'targets' : []
            }

            for G in grounding_set:
                # NOTE: the target key will instead hold the name of the anchor in question:
                anchor_name = G['target'].pop()
                output['targets'].append(get_target_position(relation, anchor_name, sre=sre))

            spg_output.append(output)

        else:
            # NOTE: this means we only have a single target and either:
            #   1. a single anchor (e.g., <tgt> left of <anc1>)
            #   2. two anchors (e.g., <tgt> between <anc1> and <anc2>):

            is_valid = False

            output = {
                'sre' : sre,
                'groundings': []
            }

            for G in grounding_set:
                target_name = G['target'].pop()
                anchor_names = []
                for A in range(len(G['anchor'])):
                    anchor_names.append(G['anchor'][A])

                is_valid = evaluate_spg(relation, target_name, anchor_names, sre=sre)

                if is_valid:
                    # print(grounding_set.index(G), ':', G['score'])
                    output['groundings'].append({
                        'target': target_name,
                        'anchor': anchor_names
                    })

                if len(output['groundings']) == topk:
                    # TODO: we are currently going by top groundings based on joint cosine similarity score;
                    #   is there some other way that can weigh both distance of target and the joint score?
                    # print(output['groundings'])
                    break

            spg_output.append(output)

        plt.close('all')

    return spg_output
#enddef

if __name__ == '__main__':

    # -- just run this part with the output from regular expression grounding (REG):
    init()
    reg_outputs = load_from_file(reg_output_path)
    for reg_output in reg_outputs:
        spg(reg_output)
