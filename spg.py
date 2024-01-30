import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json

from spot.load_map import load_map

# NOTE: spot_waypoints_path :- path to the folder of files generated by GraphNav:
spot_graph_path = f'{sys.path[0]}/spot/graphs/blackstone/'

# NOTE: osm_path :- file path for the JSON file containing OSM-extracted landmarks:
osm_path = f'{sys.path[0]}/data/osm/blackstone.json'

# NOTE: reg_output_path :- file path to the output of the RER process (obtained from Jason):
reg_output_path = f'{sys.path[0]}/data/reg_outs_blackstone.json'

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

anchor_landmarks, target_landmarks = None, None

use_pyproj = False

# -- let's assume that we will only look for an object that is within 10m of the anchor:
max_range = 10.0
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

def align_coordinates(osm_landmarks, spot_waypoints, grounding_landmark):

    global use_pyproj

    use_pyproj = True

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

    print(np.rad2deg(dir_robot))
    print(np.rad2deg(dir_world))

    angle_diff = dir_world - dir_robot

    # -- use the name of the images taken at each waypoint to indicate the actual waypoints of interest:
    global spot_waypoints_path
    list_waypoints = os.listdir(spot_graph_path + '/images/')

    target_landmarks = {}

    for W in list_waypoints:
        # -- each image is named after the Spot waypoint name (auto-generated by GraphNav):
        id_name = os.path.splitext(os.path.basename(W))[0]

        # -- just get the x-coordinate and y-coordinate, which would correspond to a top-view 2D map:
        spot_coordinate = np.array([spot_waypoints[id_name].waypoint_tform_ko.position.x,
                                   spot_waypoints[id_name].waypoint_tform_ko.position.y])

        # -- align the Spot's coordinates to the world frame:
        spot_coordinate = np.dot(rotation_matrix(angle=angle_diff), spot_coordinate)

        # -- we will use the newly rotated point to figure out the offset:
        if id_name == grounding_landmark[0]['waypoint']:
            known_waypoint_1 = spot_coordinate
        elif id_name == grounding_landmark[1]['waypoint']:
            known_waypoint_2 = spot_coordinate

        target_landmarks[id_name] = {
            'x': spot_coordinate[0],
            'y': spot_coordinate[1],
        }

    # -- compute an offset that can be used to align the known landmark from world to Spot space:
    offset = ((known_waypoint_1 - known_landmark_1) + (known_waypoint_2 - known_landmark_2)) / 2.0 

    # -- search for the robot's default starting position:
    for W in spot_waypoints:
        if spot_waypoints[W].annotations.name == 'waypoint_0':
            # -- add the initial waypoint as the robot's starting position:
            spot_coordinate = np.array([spot_waypoints[W].waypoint_tform_ko.position.x,
                                        spot_waypoints[W].waypoint_tform_ko.position.y])

            # -- align the Spot's coordinates to the world frame:
            spot_coordinate = np.dot(rotation_matrix(angle=angle_diff), spot_coordinate)

            # -- save the robot position as a global variable in case:
            global robot
            robot = {
                'x': spot_coordinate[0],
                'y': spot_coordinate[1],
            }

            break

    anchor_landmarks = {}

    for L in osm_landmarks:
        # -- we first need to convert each point into its Cartesian equivalent, then add the computed offset from above:
        if use_pyproj:
            landmark_cartesian = np.array(llh_to_xyz.transform(osm_landmarks[L]['long'], osm_landmarks[L]['lat'], 30, radians=False)[:-1])
        else:
            landmark_cartesian = np.array([gps_to_cartesian(osm_landmarks[L])[x] for x in [0, 2]])

        landmark_cartesian += offset

        # -- replace whitespace with underscore, make lowercase:
        id_name = str(L).lower().replace(' ', '_')

        anchor_landmarks[id_name] = {
            'x': landmark_cartesian[0],
            'y': landmark_cartesian[1],
        }

    return anchor_landmarks, target_landmarks
#enddef


def compute_area(spatial_rel, anchor_position, do_360_search=False, plot=False):
    list_ranges = []

    # NOTE: we want to draw a vector from the anchor's perspective to the robot!
    # -- this gives us a normal vector pointing outside of the anchor object

    # -- compute vector between robot's position and anchor position and get its direction:
    vector_a2r = [robot['x'] - anchor_position['x'],
                    robot['y'] - anchor_position['y']]

    # -- draw a unit vector and multiply it by 10 to get the max distance to consider:
    unit_vec_a2r = np.array(vector_a2r) / np.linalg.norm(vector_a2r)

    # NOTE: mean angle of 0 if we get the spatial relation "in front of" or "opposite"
    mean_angle = 0
    if spatial_rel in ['left', 'left of']:
        # -- if we want something to the left, we need to go in positive 90 degrees:
        mean_angle = -90
    elif spatial_rel in ['right', 'right of']:
        # -- if we want something to the right, we need to go in negative 90 degrees:
        mean_angle = 90
    elif spatial_rel in ['behind', 'at the rear of', 'behind of']:
        # -- if we want something to the right, we need to tn 180 degees:
        mean_angle = 180
    elif spatial_rel in ['north of', 'south of', 'east of', 'west of', 'northeast of', 'northwest of', 'southeast of', 'southwest of']:
        # -- we need to find the difference between each cardinal direction and the current anchor-to-robot vector
        #       to figure out how much we need to rotate it by:
        if spatial_rel in ['north']:
            mean_angle = np.rad2deg(np.arctan2(
                1, 0) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
        if spatial_rel in ['south']:
            mean_angle = np.rad2deg(
                np.arctan2(-1, 0) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
        if spatial_rel in ['east']:
            mean_angle = np.rad2deg(np.arctan2(
                0, 1) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
        if spatial_rel in ['west']:
            mean_angle = np.rad2deg(np.arctan2(
                0, -1) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
        if spatial_rel in ['northeast']:
            mean_angle = np.rad2deg(np.arctan2(
                1, 1) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
        if spatial_rel in ['northwest']:
            mean_angle = np.rad2deg(np.arctan2(
                1, -1) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
        if spatial_rel in ['southeast']:
            mean_angle = np.rad2deg(
                np.arctan2(-1, 1) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))
        if spatial_rel in ['southwest']:
            mean_angle = np.rad2deg(
                np.arctan2(-1, -1) - np.arctan2(unit_vec_a2r[1], unit_vec_a2r[0]))

        # NOTE: since cardinal directions are absolute, we should not do any 360-sweep:
        do_360_search = False

    # endif

    # -- checking for sweep condition: this means we will consider different normal vectors
    #       representing the "front" of the object:
    rot_a2r = [0]
    if spatial_rel in ['near', 'near to', 'next', 'next to', 'adjacent to', 'close to'] or do_360_search:
        rot_a2r += [x * 90 for x in range(1, 4)]

    # print(rot_a2r)
    for x in rot_a2r:
        # -- rotate the anchor's frame of reference by some angle x:
        a2r_vector = np.dot(rotation_matrix(
            angle=np.deg2rad(x)), unit_vec_a2r)

        # -- compute the mean vector as well as vectors representing min and max proximity range:
        a2r_mean = np.dot(rotation_matrix(
            angle=np.deg2rad(mean_angle)), a2r_vector)
        a2r_min_range = np.dot(rotation_matrix(
            angle=np.deg2rad(mean_angle-45)), a2r_vector)
        a2r_max_range = np.dot(rotation_matrix(
            angle=np.deg2rad(mean_angle+45)), a2r_vector)

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
        plt.scatter(x=[anchor_position['x']], y=[anchor_position['y']], marker='o', color='orange', label='anchor')
        plt.text(anchor_position['x'], anchor_position['y'], s=anchor_position['name'])

        plt.plot([robot['x'], anchor_position['x']], [
                    robot['y'], anchor_position['y']], color='black')
        plt.arrow(x=robot['x'], y=robot['y'], dx=-vector_a2r[0]/2.0, dy=-vector_a2r[1]/2.0, shape='full',
                    width=0.01, head_width=0.1, color='black', label='normal')

        for r in range(len(list_ranges)):
            mean_pose = [(list_ranges[r]['mean'][0] * max_range) + anchor_position['x'],
                            (list_ranges[r]['mean'][1] * max_range) + anchor_position['y']]
            plt.scatter(x=[mean_pose[0]], y=[mean_pose[1]],
                        c='g', marker='o', label=f'mean_{r}')

            min_pose = [(list_ranges[r]['min'][0] * max_range) + anchor_position['x'],
                        (list_ranges[r]['min'][1] * max_range) + anchor_position['y']]
            plt.scatter(x=[min_pose[0]], y=[min_pose[1]],
                        c='r', marker='x', label=f'min_{r}')

            max_pose = [(list_ranges[r]['max'][0] * max_range) + anchor_position['x'],
                        (list_ranges[r]['max'][1] * max_range) + anchor_position['y']]
            plt.scatter(x=[max_pose[0]], y=[max_pose[1]],
                        c='b', marker='x', label=f'max_{r}')

            plt.plot([anchor_position['x'], mean_pose[0]], [
                        anchor_position['y'], mean_pose[1]], linestyle='dashed', c='g')
            plt.plot([anchor_position['x'], min_pose[0]], [
                        anchor_position['y'], min_pose[1]], linestyle='dotted', c='r')
            plt.plot([anchor_position['x'], max_pose[0]], [
                        anchor_position['y'], max_pose[1]], linestyle='dotted', c='b')
        # endfor

        plt.legend()
        plt.axis('square')
        plt.show(block=False)

    return list_ranges
# enddef


def evaluate_spg(spatial_rel, target_candidate, anchor_candidates, do_360_search=False):

    global robot, anchor_landmarks, target_landmarks, max_range

    # -- in this case, we will be given a list of target objects or entities:
    target = target_landmarks[target_candidate]

    for A in anchor_candidates:
        
        try:
            anchor = anchor_landmarks[A]
        except KeyError:
            # TODO: what if a landmark doesn't exist as an anchor? Then we should use the target landmark?
            # input('LA')
            continue

        anchor['name'] = A

        list_ranges = compute_area(spatial_rel, anchor)

        for R in list_ranges:
            dir_tgt = np.arctan2(target['y'] - anchor['y'], target['x'] - anchor['x'])
            dir_min = np.arctan2(R['min'][1], R['min'][0])
            dir_max = np.arctan2(R['max'][1], R['max'][0])

            distance_a2t = np.linalg.norm(np.array([target['x'], target['y']]) - np.array([anchor['x'], anchor['y']]))

            if dir_tgt >= dir_min and dir_tgt <= dir_max and distance_a2t < max_range:
                print('YES!')

                # -- plot the computed range:
                plt.figure()
                plt.title(f'Evaluated range for spatial relation "{spatial_rel}"')
                plt.scatter(x=[robot['x']], y=[robot['y']], marker='o', color='yellow', label='robot')
                plt.scatter(x=[anchor['x']], y=[anchor['y']], marker='o', color='orange', label='anchor')
                plt.scatter(x=[target['x']], y=[target['y']], marker='o', color='green', label='target')
                plt.text(anchor['x'], anchor['y'], s=A)
                plt.text(target['x'], target['y'], s=target_candidate)

                mean_pose = np.array([(R['mean'][0] * max_range) + anchor['x'],
                                (R['mean'][1] * max_range) + anchor['y']])
                # plt.scatter(x=[mean_pose[0]], y=[mean_pose[1]], c='g', marker='o', label='mean')

                min_pose = np.array([(R['min'][0] * max_range) + anchor['x'],
                            (R['min'][1] * max_range) + anchor['y']])
                # plt.scatter(x=[min_pose[0]], y=[min_pose[1]], c='r', marker='x', label='min')

                max_pose = np.array([(R['max'][0] * max_range) + anchor['x'],
                            (R['max'][1] * max_range) + anchor['y']])
                # plt.scatter(x=[max_pose[0]], y=[max_pose[1]], c='b', marker='x', label='max')

                plt.plot([anchor['x'], mean_pose[0]], [anchor['y'], mean_pose[1]], linestyle='dotted', c='g')
                plt.plot([anchor['x'], min_pose[0]], [anchor['y'], min_pose[1]], linestyle='dotted', c='r')
                plt.plot([anchor['x'], max_pose[0]], [anchor['y'], max_pose[1]], linestyle='dotted', c='b')

                plt.legend()
                plt.axis('square')
                plt.show(block=False)

                return True
        
    # endif
    return False
# enddef


def get_target_position(spatial_rel, anchor_candidate):
    # -- this means that we have no target landmark: we solely want to find a position relative to a given anchor
    try:
        anchor = anchor_landmarks[anchor_candidate]
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

    plt.figure()
    plt.scatter(x=[robot['x']], y=[robot['y']], marker='o', label='robot')
    plt.scatter(x=[new_robot_pos['x']], y=[new_robot_pos['y']], marker='o', c='g', label='new robot pose')

    # -- plot all anchors and targets provided to the function:
    for A in anchor_landmarks:
        plt.scatter(x=anchor_landmarks[A]['x'], y=anchor_landmarks[A]['y'],
                    marker='o', c='darkorange',
                    label=f"anchor: {A}")
        plt.text(anchor_landmarks[A]['x'], anchor_landmarks[A]['y'], A)

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


def plot_landmarks(anchor_landmarks=None, target_landmarks=None):
    # -- plotting the points in the shared world space local to the Spot's map:
    plt.figure()
    if anchor_landmarks:
        plt.scatter(x=[anchor_landmarks[A]['x'] for A in anchor_landmarks], y=[anchor_landmarks[A]['y'] for A in anchor_landmarks], c='orange', label='anchor landmarks')
        for A in anchor_landmarks:
            plt.text(anchor_landmarks[A]['x'], anchor_landmarks[A]['y'], A)
    
    if target_landmarks:
        plt.scatter(x=[target_landmarks[T]['x'] for T in target_landmarks], y=[target_landmarks[T]['y'] for T in target_landmarks], c='green', label='target landmarks')
        for T in target_landmarks:
            plt.text(target_landmarks[T]['x'], target_landmarks[T]['y'], T)
    
    if robot:
        plt.scatter(x=robot['x'], y=robot['y'], c='yellow', label='robot')
    
    plt.title('Landmarks: Target and Anchor')
    plt.legend()
    plt.axis('square')
    plt.show(block=False)
#enddef


def init(osm_landmark_file=None):
    global anchor_landmarks, target_landmarks

    # -- load the waypoints recorded using Spot's GraphNav:
    (_, waypoints, _, _, _, _) = load_map(spot_graph_path)

    if osm_landmark_file:
        osm_path = osm_landmark_file

    # -- load the OSM landmarks from a JSON file:
    osm_landmarks = json.load(open(osm_path, 'r'))

    # -- iterate through all of the Spot waypoints as well as the OSM landmarks and put them in the same space:
    anchor_landmarks, target_landmarks = align_coordinates(
        osm_landmarks, waypoints, grounding_landmark=grounding_landmark)
#enddef


def spg(spatial_preds):

    global anchor_landmarks, target_landmarks

    ########################################################################################################
    # # NOTE: Sanity check for trying to create a top-down map of the world:
    # pvd = {'lat':41.816909,'long':-71.5768957,'name':'PVD'}
    # paris = {'lat':48.8589383,'long':2.2644633, 'name':'PARIS'}
    # cape_town = {'lat':-33.9145272,'long':18.3257368, 'name': 'CAPE'}
    # beijing = {'lat':39.9389417,'long':116.0671357, 'name': 'BEI'}
    # monte = {'lat':-34.8178097,'long':-56.850685, 'name': 'MONT'}

    # plt.figure()
    # for M in [pvd, paris, cape_town, beijing, monte]:
    #     coord = Transformation.gps_to_cartesian(M)
    #     plt.scatter(x=coord[0], y=coord[2], label=M['name'])
    # plt.legend()
    # plt.show()
    ########################################################################################################

    # -- plot the points for visualization purposes:
    plot_landmarks(anchor_landmarks, target_landmarks)

    # -- find the closest waypoint to each anchor from OSM:
    anchor_to_target = {}
    for A in anchor_landmarks:
        this_anchor = np.array([anchor_landmarks[A]['x'], anchor_landmarks[A]['y']])

        closest_waypoint = None
        for target in target_landmarks:
            if not closest_waypoint:
                closest_waypoint = target
                best_waypoint = np.array([target_landmarks[closest_waypoint]['x'], target_landmarks[closest_waypoint]['y']])
            else:
                # -- check if the current target (waypoint) is closer than the closest target stored as a variable:
                this_waypoint = np.array([target_landmarks[target]['x'], target_landmarks[target]['y']])
                best_waypoint = np.array([target_landmarks[closest_waypoint]['x'], target_landmarks[closest_waypoint]['y']])

                # NOTE: closest in terms of Euclidean distance:
                if np.linalg.norm(this_anchor-this_waypoint) <= np.linalg.norm(this_anchor-best_waypoint):
                    closest_waypoint = target

        # NOTE: the best target will be the waypoint closest to the anchor:
        anchor_to_target[A] = closest_waypoint
        # print(A, anchor_to_target[A])

    for R in range(len(spatial_preds['grounded_spatial_preds'])):
        reg_dict = spatial_preds['grounded_spatial_preds'][R]
        print(spatial_preds['sres'][R])

        # NOTE: the spatial relation is always the first key:
        relation = list(reg_dict.keys()).pop()

        if relation == 'between':
            continue

        if len(reg_dict[relation]) == 1:
            # -- this means we only have an anchoring landmark and no target landmark:
            anchor = reg_dict[relation][0]

            target_positions = []

            for A in range(len(anchor)):
                anchor_id = anchor[A][1]
                print(anchor_id)
                target_positions.append(get_target_position(relation, anchor_id))

        else:
            # -- this means we only have a single target and either:
            #   1. a single anchor (e.g., <tgt> left of <anc1>) 
            #   2. two anchors (e.g., <tgt> between <anc1> and <anc2>):
            target, anchor_1, anchor_2 = reg_dict[relation][0], reg_dict[relation][1], reg_dict[relation][2] if len(reg_dict[relation]) > 2 else None

            is_valid = False

            for T in range(len(target)):
                # -- we are evaluating candidates for targets based on the REG confidence score:
                for A in range(len(anchor_1)):
                    target_id, anchor_1_id, anchor_2_id = target[T][1], anchor_1[A][1], anchor_2[A][1] if anchor_2 else None
                    print(target_id, anchor_1_id, anchor_2_id)
                    is_valid = evaluate_spg(relation, target_id, [anchor_1_id, anchor_2_id] if anchor_2 else [anchor_1_id])

                    if is_valid:
                        return {
                            'target': target_id,
                            'anchor': [anchor_1_id, anchor_2_id] if anchor_2 else [anchor_1_id]
                        }

    return None

if __name__ == '__main__':
    # -- just run this part with the output from regular expression grounding (REG):
    init()
    reg_outputs = json.load(open(reg_output_path, 'r'))
    for reg_output in reg_outputs:
        spg(reg_output)
