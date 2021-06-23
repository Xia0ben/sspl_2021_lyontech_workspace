# -*- coding: utf-8 -*-

import actionlib
import cv2
import glob
import math
import moveit_commander
import numpy as np
import os
import rospy
import ros_numpy
import subprocess
import tf
import tf2_ros
import time
import json
import threading
import copy
from sklearn.cluster import DBSCAN
import traceback
from future.utils import with_metaclass
from shapely.geometry import Polygon, MultiPoint, Point
from shapely import affinity
import matplotlib.pyplot as plt
from aabbtree import AABB, AABBTree


from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped, Twist, PointStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionGoal
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Header, String


from rospy_message_converter import message_converter


SAVED_RIGHT_WALL_COORDS = [(3.0918407440185547, -0.388437420129776), (3.091840982437134, 4.971062660217285), (3.2908992767333984, 4.962766647338867), (3.2328405380249023, -0.4050302505493164)]
SAVED_LEFT_WALL_COORDS = [(-0.6239157915115356, 2.216644763946533), (2.079960823059082, 2.200051784515381), (2.088254928588867, 2.0258266925811768), (-0.6239157915115356, 2.0092337131500244)]
SAVED_TABOO_AREA = [(2.0965490341186523, 1.9096763134002686), (2.784959316253662, 2.0175302028656006), (2.8015475273132324, 3.203921318054199), (2.079960823059082, 3.2868852615356445)]

ROBOT_POLYGON_FOR_SWIPING = Polygon(
[(0.1857410515735669, 0.16458913801365993),
 (0.5765513089663079, 0.1475914923236168),
 (0.568078760730122, 0.01280330703197885),
 (0.24559342643772342, -0.01245386605347587),
 (0.22272638886728996, -0.10838118345469905),
 (0.16665210547897402, -0.18637543528837552),
 (0.03952898328804627, -0.25494984341738824),
 (-0.09375177029448534, -0.23765193264520423),
 (-0.19616346866617784, -0.16519818795399166),
 (-0.23428419005300327, -0.08886792653772968),
 (-0.2537417481254019, -0.00975102918455395),
 (-0.24138615665000795, 0.060762705251869775),
 (-0.21242668082254168, 0.122310708272241),
 (-0.15789661578754483, 0.19149136853856996),
 (-0.07118869751963597, 0.2368641886049927),
 (0.038730715358997136, 0.24182999298870067),
 (0.08766237530813098, 0.22719946648419231),
 (0.14819992055765457, 0.1923655289839865)])

RIGHT_WALL_POLYGON = Polygon(SAVED_RIGHT_WALL_COORDS)
LEFT_WALL_POLYGON = Polygon(SAVED_LEFT_WALL_COORDS)
TABOO_AREA_POLYGON = Polygon(SAVED_TABOO_AREA)

OTHER_POLYGONS = {1: RIGHT_WALL_POLYGON, 2: LEFT_WALL_POLYGON}

def get_current_time_sec():
    return rospy.Time.now().to_sec()


def quaternion_from_euler(roll, pitch, yaw):
    q = tf.transformations.quaternion_from_euler(
        math.radians(roll), math.radians(pitch), math.radians(yaw), 'rxyz'
    )
    return Quaternion(q[0], q[1], q[2], q[3])


def angle_to_360_interval(angle):
    final_angle = angle % 360.
    final_angle = final_angle if final_angle >= 0. else final_angle + 360.
    return final_angle


def get_translation(start_pose, end_pose):
    return end_pose[0] - start_pose[0], end_pose[1] - start_pose[1]


def get_rotation(start_pose, end_pose):
    return angle_to_360_interval(end_pose[2] - start_pose[2])


def get_translation_and_rotation(start_pose, end_pose):
    translation = get_translation(start_pose, end_pose)
    rotation = get_rotation(start_pose, end_pose)
    return translation, rotation


def rotate_then_translate_polygon(polygon, translation, rotation, rotation_center='center'):
    return affinity.translate(affinity.rotate(polygon, rotation, origin=rotation_center), *translation)


def set_polygon_pose(polygon, init_polygon_pose, end_polygon_pose, rotation_center='center'):
    translation, rotation = get_translation_and_rotation(init_polygon_pose, end_polygon_pose)
    return rotate_then_translate_polygon(polygon, translation, rotation, rotation_center)


def euclidean_distance(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)





class Action:

    def __init__(self):
        pass


class Rotation(Action):

    def __init__(self, angle, center):
        Action.__init__(self)
        self.angle = angle
        self.center = center

    def apply(self, polygon):
        return affinity.rotate(geom=polygon, angle=self.angle, origin=self.center, use_radians=False)


class Translation(Action):

    def __init__(self, translation_vector):
        Action.__init__(self)
        self.translation_vector = translation_vector

    def apply(self, polygon):
        return affinity.translate(geom=polygon, xoff=self.translation_vector[0], yoff=self.translation_vector[1], zoff=0.)


def bounds(points):
    minx, miny, maxx, maxy = float("inf"), float("inf"), -float("inf"), -float("inf")
    for point in points:
        minx, miny, maxx, maxy = min(minx, point[0]), min(miny, point[1]), max(maxx, point[0]), max(maxy, point[1])
    return minx, miny, maxx, maxy


def rotate(point, angle, center, radius=None, radians=False):
    if not radius:
        radius = math.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
    if not radians:
        angle = math.radians(angle)
    angle += math.atan2((point[1] - center[1]), (point[0] - center[0]))
    return center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)


def arc_bounding_box(point_a, rot_angle, center, point_b=None, point_c=None, bb_type='minimum_rotated_rectangle'):
    """
    Computes the bounding box of the arc formed by the rotation of a point A around a given center
    :param point_a: Initial point state
    :type point_a: (float, float)
    :param rot_angle: rotation angle in degrees.
    :type rot_angle: float
    :param center: rotation origin point
    :type center: (float, float)
    :param point_b: Final point state after rotation, can be provided to accelerate computation
    :type point_b: (float, float)
    :param point_c: Middle point state after rotation, can be provided to accelerate computation
    :type point_c: (float, float)
    :param bb_type: Type of bounding box, either 'minimum_rotated_rectangle' or 'aabbox', first one is most accurate
    :type bb_type: str
    :return: Return a list of four points coordinates corresponding to the bounding box
    :rtype: [(float, float), (float, float), (float, float), (float, float)]
    """
    if not point_b:
        r = math.sqrt((point_a[0] - center[0]) ** 2 + (point_a[1] - center[1]) ** 2)
        point_b = rotate(point_a, rot_angle, center, radius=r)
    else:
        r = None

    if -1.e-15 < rot_angle < 1.e-15:
        # It means that there is no movement, return only A
        return [point_a]
    elif -180. <= rot_angle <= 180.:
        # If the arc is less than a half circle

        # Compute middle point C
        if not point_c:
            point_c = rotate(point_a, rot_angle / 2., center)

        if bb_type is 'minimum_rotated_rectangle':
            # The minimum rotated rectangle's corners are points A, B, D and E
            # D and E are the intersection points between the line parallel to [AB] passing by C, and respectively,
            # the lines perpendicular to [AB] passing by A and B.
            x_b_min_a, y_b_min_a = (point_b[0] - point_a[0]), (point_b[1] - point_a[1])
            if -1.e-15 < x_b_min_a < 1.e-15:
                # Special case where [AB] is vertical
                point_d, point_e = (point_c[0], point_a[1]), (point_c[0], point_b[1])
            else:
                # General case
                m_ab = y_b_min_a / x_b_min_a  # [AB]'s slope = [DC]'s slope
                if -1.e-15 < m_ab < 1.e-15:
                    # Special case where [AB] is horizontal
                    point_d, point_e = (point_a[0], point_c[1]), (point_b[0], point_c[1])
                else:
                    b_dc = point_c[1] - m_ab * point_c[0]
                    m_ad = 0. if m_ab >= 1e15 else -1. / m_ab
                    b_ad = point_a[1] - m_ad * point_a[0]
                    xd = (b_ad - b_dc) / (m_ab - m_ad)
                    yd = xd * m_ab + b_dc
                    point_d = (xd, yd)
                    # C is the midpoint between D and E, allowing us to compute E
                    point_e = (2. * point_c[0] - point_d[0], 2. * point_c[1] - point_d[1])
            return [point_a, point_b, point_d, point_e]
        elif bb_type is 'aabbox':
            # The aabb corners are simply the bounds of points A, B and C.
            minx, miny, maxx, maxy = bounds([point_a, point_b, point_c])
            return [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
    elif -360. < rot_angle < 360.:
        # If the arc is greater than a half circle but not a circle
        # then we have 5 extremal points : A, B, C, D and E.
        # C is the arc middle point
        # D and E are the intersection points between the circle's equation and the ray that is perpendicular
        # to the ray passing through C

        # Compute middle point C and the radius if not already computed
        if not r:
            r = math.sqrt((point_a[0] - center[0]) ** 2 + (point_a[1] - center[1]) ** 2)
        if not point_c:
            point_c = rotate(point_a, rot_angle / 2., center, radius=r)

        # Compute the slope of the ray passing through C
        m1 = (point_c[1] - center[1]) / (point_c[0] - center[0])

        if -1.e-15 < m1 < 1.e-15:
            # If the ray passing through C IS horizontal

            # Line terms of the ray that is perpendicular to the ray passing through C (x=p2 is vertical line equation)
            p2 = center[0]

            # Terms of the equation to solve for x coordinate of points D and E
            a = 1.
            b = -2. * center[1]
            c = center[0] ** 2 + center[1] ** 2 + p2 ** 2 - 2. * center[0] * p2 - r ** 2

            # Solve the equation to get the coordinates of points D and E
            discriminant = (b ** 2) - (4 * a * c)

            yd = (-b - math.sqrt(discriminant)) / (2 * a)
            ye = (-b + math.sqrt(discriminant)) / (2 * a)

            xd = center[0]
            xe = center[0]

            point_d, point_e = (xd, yd), (xe, ye)

            # Now simply return the proper bounding box englobing A, B, C, D and E
            bb_points_x = [
                point_c[0],
                point_c[0],
                point_a[0],
                point_a[0]
            ]
            bb_points_y = [
                point_d[1],
                point_e[1],
                point_e[1],
                point_d[1]
            ]
            if bb_type is 'minimum_rotated_rectangle':
                return list(zip(bb_points_x, bb_points_y))
            elif bb_type is 'aabbox':
                minx, miny, maxx, maxy = bounds(list(zip(bb_points_x, bb_points_y)))
                return [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
        else:
            # If the ray passing through C is not horizontal (GENERAL CASE)

            # Line terms of the ray that is perpendicular to the ray passing through C
            m2 = 0. if m1 >= 1e15 else -1. / m1  # If ray passing through C is vertical, perpendicular is horizontal
            p2 = center[1] - m2 * center[0]

            # Terms of the equation to solve for x coordinate of points D and E
            a = 1. + m2 ** 2
            b = m2 * (2. * p2 - 2. * center[1]) - 2. * center[0]
            c = center[0] ** 2 + p2 ** 2 + center[1] ** 2 - 2. * p2 * center[1] - r ** 2

            # Solve the equation to get the coordinates of points D and E
            discriminant = (b ** 2) - (4. * a * c)

            xd = (-b - math.sqrt(discriminant)) / (2. * a)
            xe = (-b + math.sqrt(discriminant)) / (2. * a)

            yd = xd * m2 + p2
            ye = xe * m2 + p2

            point_d, point_e = (xd, yd), (xe, ye)

            # Now simply return the proper bounding box englobing A, B, C, D and E
            m_lc = m2
            p_lc = point_c[1] - m_lc * point_c[0]

            m_ld = m1
            p_ld = point_d[1] - m_ld * point_d[0]

            m_le = m1
            p_le = point_e[1] - m_le * point_e[0]

            m_lab = m2
            p_lab = point_a[1] - m_lab * point_a[0]

            bb_points_x = [
                (p_lc - p_ld) / (m_ld - m_lc),
                (p_lc - p_le) / (m_le - m_lc),
                (p_lab - p_le) / (m_le - m_lab),
                (p_lab - p_ld) / (m_ld - m_lab)
            ]
            bb_points_y = [
                m_lc * bb_points_x[0] + p_lc,
                m_lc * bb_points_x[1] + p_lc,
                m_lab * bb_points_x[2] + p_lab,
                m_lab * bb_points_x[3] + p_lab
            ]
            if bb_type is 'minimum_rotated_rectangle':
                return list(zip(bb_points_x, bb_points_y))
            elif bb_type is 'aabbox':
                minx, miny, maxx, maxy = bounds(list(zip(bb_points_x, bb_points_y)))
                return [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
    else:
        # Beyond 360 degrees, the arc is a circle: its bounding box is necessarily a square aabb
        r = math.sqrt((point_a[0] - center[0]) ** 2 + (point_a[1] - center[1]) ** 2)
        return [
            (center[0] - r, center[1] - r), (center[0] + r, center[1] - r),
            (center[0] + r, center[1] + r), (center[0] - r, center[1] + r)
        ]


def bounding_boxes_vertices(action_sequence, polygon_sequence, bb_type='minimum_rotated_rectangle'):
    """
    Returns for each action the pointclouds of the bounding boxes that cover each polygon's point trajectory
    during the action.
    :param action_sequence:
    :type action_sequence:
    :param polygon_sequence:
    :type polygon_sequence:
    :param bb_type: Type of bounding box, either 'minimum_rotated_rectangle' or 'aabbox', first one is most accurate
    :type bb_type: str
    :return:
    :rtype:
    """
    bb_vertices = []
    for index, action in enumerate(action_sequence):
        init_poly_coords = list(polygon_sequence[index].exterior.coords)
        end_poly_coords = list(polygon_sequence[index + 1].exterior.coords)
        action_bb_vertices = []
        if isinstance(action, Translation):
            for coord in init_poly_coords:
                action_bb_vertices.append(coord)
            for coord in end_poly_coords:
                action_bb_vertices.append(coord)
        elif isinstance(action, Rotation):
            for point_a, point_b in zip(init_poly_coords, end_poly_coords):
                bb = arc_bounding_box(point_a=point_a, point_b=point_b, rot_angle=action.angle, center=action.center, bb_type=bb_type)
                for coord in bb:
                    action_bb_vertices.append(coord)
        else:
            raise TypeError("Actions must be pure Translation or Rotation.")
        bb_vertices.append(action_bb_vertices)
    return bb_vertices


def csv_from_bb_vertices(bb_vertices):
    """
    Computes the CSV (Convex Swept Volume) approximation polygon of the provided bounding boxes vertices
    :param bb_vertices: List of Bounding boxes vertices for each action
    :type bb_vertices:
    :return: The CSV (Convex Swept Volume) approximation polygon
    :rtype: shapely.geometry.Polygon
    """
    all_vertices = [vertex for vertices in bb_vertices for vertex in vertices]
    return MultiPoint(all_vertices).convex_hull


def polygon_to_aabb(polygon):
    xmin, ymin, xmax, ymax = polygon.bounds
    return AABB([(xmin, xmax), (ymin, ymax)])


def polygons_to_aabb_tree(polygons):
    aabb_tree = AABBTree()
    for uid, polygon in polygons.items():
        aabb_tree.add(polygon_to_aabb(polygon), uid)
    return aabb_tree


def check_static_collision(main_uid, polygon, other_entities_polygons, aabb_tree, ignored_uids=None, break_at_first=True, save_intersections=False):
    aabb = polygon_to_aabb(polygon)
    potential_collision_uids = aabb_tree.overlap_values(aabb)
    if ignored_uids:
        potential_collision_uids = set(potential_collision_uids).difference(set(ignored_uids))
    if break_at_first:
        for uid in potential_collision_uids:
            if polygon.intersects(other_entities_polygons[uid]):
                if save_intersections:
                    intersection = polygon.intersection(other_entities_polygons[uid])
                    return {main_uid: {uid}, uid: {main_uid}}, {(main_uid, uid): intersection, (uid, main_uid): intersection}
                else:
                    return {main_uid: {uid}, uid: {main_uid}}
        return {}
    else:
        collides_with = {}
        if save_intersections:
            intersections = {}
        for uid in potential_collision_uids:
            if polygon.intersects(other_entities_polygons[uid]):
                if save_intersections:
                    intersection = polygon.intersection(other_entities_polygons[uid])
                    intersections[(main_uid, uid)] = intersection
                    intersections[(uid, main_uid)] = intersection

                if main_uid in collides_with:
                    collides_with[main_uid].add(uid)
                else:
                    collides_with[main_uid] = {uid}

                if uid in collides_with:
                    collides_with[uid].add(main_uid)
                else:
                    collides_with[uid] = {main_uid}

        if save_intersections:
            return collides_with, intersections
        else:
            return collides_with


def merge_collides_with(source, other):
    for uid, uids in other.items():
        if uid in source:
            source[uid].update(uids)
            for uid_2 in uids:
                if uid_2 in source:
                    source[uid_2].add(uid)
                else:
                    source[uid_2] = {uid}
        else:
            source[uid] = uids
            for uid_2 in uids:
                if uid_2 in source:
                    source[uid_2].add(uid)
                else:
                    source[uid_2] = {uid}
    return source


def csv_check_collisions(main_uid, other_polygons, polygon_sequence, action_sequence, id_sequence=None,
                         bb_type='minimum_rotated_rectangle', aabb_tree=None, bb_vertices=None, csv_polygons=None,
                         intersections=None, ignored_entities=None, display_debug=False, break_at_first=True,
                         save_intersections=False):
    # Initialize at first recursive iteration
    if not aabb_tree:
        aabb_tree = polygons_to_aabb_tree(other_polygons)
    if not bb_vertices:
        bb_vertices = bounding_boxes_vertices(action_sequence, polygon_sequence, bb_type)
    if not csv_polygons:
        csv_polygons = {}
    if not intersections:
        intersections = {}
    if not id_sequence:
        id_sequence = range(len(action_sequence))

    csv_polygon = csv_from_bb_vertices(bb_vertices)
    csv_polygons[tuple(id_sequence)] = csv_polygon

    # Dichotomy-check for collision between polygon and CSV as long as:
    # - there is no collision
    # - AND the CSV envelops more than one action (two consecutive polygons)
    if save_intersections:
        collides_with, local_intersections = check_static_collision(
            main_uid, csv_polygon, other_polygons, aabb_tree, ignored_entities, break_at_first, save_intersections
        )
        intersections[tuple(id_sequence)] = local_intersections
    else:
        collides_with = check_static_collision(
            main_uid, csv_polygon, other_polygons, aabb_tree, ignored_entities, break_at_first, save_intersections
        )

    if collides_with:
        if display_debug:
            fig, ax = plt.subplots()
            for p in polygon_sequence:
                ax.plot(*p.exterior.xy, color='grey')
            # for i in indexes:
            #     ax.plot(*polygon_sequence[i].exterior.xy, color='blue')
            for p in other_polygons.values():
                ax.plot(*p.exterior.xy, color='black')
            x, y = zip(*[[vertex.x, vertex.y] for vertex in bb_vertices])
            ax.scatter(x, y, marker='x')
            ax.plot(*csv_polygon.exterior.xy, color='green')
            intersection = csv_polygon.intersection(other_polygons[collides_with[main_uid][0]])
            ax.plot(*intersection.exterior.xy, color='red')
            ax.axis('equal')
            fig.show()
            print("")

        if len(bb_vertices) >= 2:
            first_half_bb_vertices = bb_vertices[:len(bb_vertices) // 2]
            second_half_bb_vertices = bb_vertices[len(bb_vertices) // 2:]
            first_half_ids = id_sequence[:len(id_sequence) // 2]
            second_half_ids = id_sequence[len(id_sequence) // 2:]
            first_half_collides, first_half_collides_with, _, _, _, _ = csv_check_collisions(
                main_uid, other_polygons, polygon_sequence, action_sequence, first_half_ids, aabb_tree=aabb_tree,
                bb_vertices=first_half_bb_vertices, ignored_entities=ignored_entities, display_debug=display_debug,
                break_at_first=break_at_first, bb_type=bb_type, csv_polygons=csv_polygons, intersections=intersections
            )
            second_half_collides, second_half_collides_with, _, _, _, _ = csv_check_collisions(
                main_uid, other_polygons, polygon_sequence, action_sequence, second_half_ids, aabb_tree=aabb_tree,
                bb_vertices=second_half_bb_vertices, ignored_entities=ignored_entities, display_debug=display_debug,
                break_at_first=break_at_first, bb_type=bb_type, csv_polygons=csv_polygons, intersections=intersections
            )
            collides_with = merge_collides_with(first_half_collides_with, second_half_collides_with)
            collides = first_half_collides or second_half_collides
            return collides, collides_with, aabb_tree, csv_polygons, intersections, bb_vertices
        else:
            return True, collides_with, aabb_tree, csv_polygons, intersections, bb_vertices
    else:
        return False, collides_with, aabb_tree, csv_polygons, intersections, bb_vertices



class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Robot(with_metaclass(Singleton)):
    JOINTS_FOR_SWIPING = [0.] + [math.radians(a) for a in [-146., 0., 53., 0., 0.]]

    def __init__(self):
        init_threads = []

        init_threads.append(threading.Thread(target=self.init_base_vel_pub))
        init_threads.append(threading.Thread(target=self.init_arm))
        init_threads.append(threading.Thread(target=self.init_gripper))
        init_threads.append(threading.Thread(target=self.init_head))
        init_threads.append(threading.Thread(target=self.init_base))
        init_threads.append(threading.Thread(target=self.init_tf_listener))
        init_threads.append(threading.Thread(target=self.init_navclient))

        for t in init_threads:
            t.start()

        for t in init_threads:
            t.join()

    def init_base_vel_pub(self):
        self.base_vel_pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=1)

    def init_arm(self):
        self.arm = moveit_commander.MoveGroupCommander('arm')

    def init_gripper(self):
        self.gripper = moveit_commander.MoveGroupCommander("gripper")

    def init_head(self):
        self.head = moveit_commander.MoveGroupCommander("head")

    def init_base(self):
        self.base = moveit_commander.MoveGroupCommander("base")

    def init_tf_listener(self):
        self.tf_listener = tf.TransformListener()

    def init_navclient(self):
        self.navclient = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
        is_mb_up = self.is_move_base_up()
        while not is_mb_up:
            rospy.logwarn("Unable to connect to Navigation Action Server, state: {}".format(is_mb_up))
            is_mb_up = self.is_move_base_up()
        rospy.loginfo("Navigation Action Server Connected, state: {}".format(is_mb_up))

    def move_base_vel(self, vx, vy, vw):
        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.angular.z = math.radians(vw)
        self.base_vel_pub.publish(twist)

    def move_base_actual_goal(self, goal, timeout=60.):
        self.navclient.send_goal(goal)
        self.navclient.wait_for_result(timeout=rospy.Duration(timeout))
        state = self.navclient.get_state()
        return True if state == 3 else False

    def move_base_goal(self, x, y, theta):
        goal = MoveBaseGoal()

        goal.target_pose.header.frame_id = "map"

        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y

        goal.target_pose.pose.orientation = quaternion_from_euler(0, 0, theta)

        self.navclient.send_goal(goal)
        self.navclient.wait_for_result()
        state = self.navclient.get_state()

        return True if state == 3 else False

    def is_move_base_up(self):
        transformed = False
        while not transformed:
            try:
                self.tf_listener.waitForTransform("map", "base_link", rospy.Time(0),rospy.Duration(2.0))
                transform = self.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
                transformed = True
            except Exception as e:
                rospy.loginfo(traceback.format_exc(e))
                time.sleep(0.1)
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.pose.position.x = transform[0][0]
        goal.target_pose.pose.position.y = transform[0][1]
        goal.target_pose.pose.position.z = transform[0][2]
        goal.target_pose.pose.orientation.x = transform[1][0]
        goal.target_pose.pose.orientation.y = transform[1][1]
        goal.target_pose.pose.orientation.z = transform[1][2]
        goal.target_pose.pose.orientation.w = transform[1][3]
        return self.move_base_actual_goal(goal, 3.)

    def get_diff_between(self, target_frame, source_frame):
        self.tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(0),rospy.Duration(4.0))
        transform = self.tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
        return transform[0][0], transform[0][1]

    def move_arm_neutral(self):
        self.arm.set_named_target('neutral')
        return self.arm.go()

    def move_arm_init(self):
        self.arm.set_named_target('go')
        return self.arm.go()

    def move_hand(self, v):
        self.gripper.set_joint_value_target("hand_motor_joint", v)
        return self.gripper.go()

    def open_hand(self):
        return self.move_hand(1)

    def close_hand(self):
        return self.move_hand(0)

    def move_head_tilt(self, v):
        self.head.set_joint_value_target("head_tilt_joint", v)
        return self.head.go()

    def move_arm_to_swiping_pose(self):
        self.arm.set_joint_value_target(self.JOINTS_FOR_SWIPING)
        return self.arm.go()


class PointsSaver:
    def __init__(self):
        self.coords = []
        self.topic_sub = rospy.Subscriber("/clicked_point", PointStamped, callback=self.save_cb)
        self.lock = threading.Lock()

    def save_cb(self, msg):
        with self.lock:
            self.coords.append((msg.point.x, msg.point.y))

    def get_coords(self):
        with self.lock:
            return str(self.coords)


class NavGoalToJsonFileSaver:
    def __init__(self, filepath="saved_msg.json"):
        self.filepath = filepath
        with open(self.filepath, "w") as f:
            json.dump({}, f)
        self.topic_sub = rospy.Subscriber("/move_base/goal", MoveBaseActionGoal, callback=self.save_cb)

    def save_cb(self, msg):
        with open(self.filepath, "w") as f:
            json.dump(message_converter.convert_ros_message_to_dictionary(msg), f)


class PixelData:
    def __init__(self, pixel, hue, x, y, z):
        self.pixel, self.hue, self.x, self.y, self.z = pixel, hue, x, y, z

    def __str__(self):
        return str(self.__dict__)


class SegmentedObject:
    def __init__(self, uid, pixels, header):
        self.uid, self.pixels, self.header = uid, pixels, header

        all_x = [pixel.x for pixel in self.pixels]
        all_y = [pixel.y for pixel in self.pixels]
        all_z = [pixel.z for pixel in self.pixels]
        all_hues = [pixel.hue for pixel in self.pixels]

        self.xyz_min = (min(all_x), min(all_y), min(all_z))
        self.xyz_max = (max(all_x), max(all_y), max(all_z))
        self.xyz_avg = (np.average(all_x), np.average(all_y), np.average(all_z))
        self.xyz_med = (np.median(all_x), np.median(all_y), np.median(all_z))

        self.hue_min = min(all_hues)
        self.hue_max = max(all_hues)
        self.hue_avg = np.average(all_hues)
        self.hue_med = np.median(all_hues)

        self.name = "object_with_hue_{}".format(self.hue_med)

    def xyz_to_pose_stamped(self, xyz):
        pose_stamped = PoseStamped(header=self.header)
        pose_stamped.pose.position.x = xyz[0]
        pose_stamped.pose.position.y = xyz[1]
        pose_stamped.pose.position.z = xyz[2]
        pose_stamped.pose.orientation = tf.transformations.quaternion_from_euler(0, 0, 0)
        return pose_stamped

    @property
    def pose_min(self):
        return xyz_to_pose_stamped(self.xyz_min)

    @property
    def pose_max(self):
        return xyz_to_pose_stamped(self.xyz_max)

    @property
    def pose_avg(self):
        return xyz_to_pose_stamped(self.xyz_avg)

    @property
    def pose_med(self):
        return xyz_to_pose_stamped(self.xyz_med)

    @property
    def bb_coords_3d(self):
        return [
            (self.xyz_min[0], self.xyz_min[1], self.xyz_min[2]),
            (self.xyz_max[0], self.xyz_min[1], self.xyz_min[2]),
            (self.xyz_max[0], self.xyz_max[1], self.xyz_min[2]),
            (self.xyz_max[0], self.xyz_max[1], self.xyz_max[2]),
            (self.xyz_min[0], self.xyz_max[1], self.xyz_max[2]),
            (self.xyz_min[0], self.xyz_min[1], self.xyz_max[2]),
            (self.xyz_max[0], self.xyz_min[1], self.xyz_max[2]),
            (self.xyz_min[0], self.xyz_max[1], self.xyz_min[2])
        ]

    @property
    def bb_coords_2d(self):
        return [
            (self.xyz_min[0], self.xyz_min[1]),
            (self.xyz_max[0], self.xyz_min[1]),
            (self.xyz_min[0], self.xyz_max[1]),
            (self.xyz_max[0], self.xyz_max[1])
        ]

class Scene(with_metaclass(Singleton)):
    FLOOR_MIN_HUE, FLOOR_MAX_HUE = 14, 30

    def __init__(self, start_on_init=True):
        self._br = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()

        self._cloud_sub = None

        self._current_objects = None
        self.lock = threading.Lock()

        self.tf_publish_freq = 10.
        self.tf_publishing_thread = threading.Thread(target=self.publish_current_objects_tfs)
        self.tf_publishing_thread.start()

        if start_on_init:
            self.start()

    def start(self):
        if not self._cloud_sub:
            with self.lock:
                self._current_objects = None
            self._cloud_sub = rospy.Subscriber(
                "/hsrb/head_rgbd_sensor/depth_registered/rectified_points",
                PointCloud2, self._cloud_cb
            )

    def pause(self):
        if self._cloud_sub:
            self._cloud_sub.unregister()
            self._cloud_sub = None

    def wait_for_one_detection(self, timeout=10., sleep_duration=0.01):
        start_time = time.time()
        self.start()
        current_objects = None
        now = time.time()
        while now - start_time < timeout:
            if self._current_objects is not None:
                current_objects = self._current_objects
            else:
                time.sleep(sleep_duration)
            now = time.time()
        self.pause()
        return current_objects

    def _cloud_cb(self, msg):
        points = ros_numpy.numpify(msg)

        image = points['rgb'].view((np.uint8, 4))[..., [2, 1, 0]]

        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
        h_image = hsv_image[..., 0]

        floor_region = (h_image > self.FLOOR_MIN_HUE) & (h_image < self.FLOOR_MAX_HUE)
        wall_and_robot_region = h_image == 0
        objects_region = wall_and_robot_region | floor_region

        object_pixels = zip(*np.where(objects_region==False))

        if not object_pixels:
            with self.lock:
                self._current_objects = {}
            return

        db_scan_result = DBSCAN(eps=4, min_samples=10).fit(object_pixels)

        uid_to_pixels = {}
        for uid, pixel in zip(db_scan_result.labels_, object_pixels):
            hue = h_image[pixel[0]][pixel[1]]
            x, y, z = (points[letter][pixel[0]][pixel[1]] for letter in ['x', 'y', 'z'])

            self.tf_listener.waitForTransform("map", msg.header.frame_id, rospy.Time(0),rospy.Duration(4.0))
            point=PointStamped()
            point.header.frame_id = msg.header.frame_id
            point.header.stamp =rospy.Time(0)
            point.point.x=x
            point.point.y=y
            point.point.z=z
            p=self.tf_listener.transformPoint("map", point)
            x_t, y_t, z_t = p.point.x, p.point.y, p.point.z

            if uid in uid_to_pixels:
                uid_to_pixels[uid].append(PixelData(pixel, hue, x_t, y_t, z_t))
            else:
                uid_to_pixels[uid] = [PixelData(pixel, hue, x_t, y_t, z_t)]

        new_objects = {uid: SegmentedObject(uid, pixels, msg.header) for uid, pixels in uid_to_pixels.items()}

        with self.lock:
            self._current_objects = new_objects
#             for new_uid, new_object in new_objects.items():
#                 for cur_uid, cur_object in self._current_objects.items():
#                     if cur_object.x

    def publish_current_objects_tfs(self):
        while not rospy.is_shutdown():
            with self.lock:
                if self._current_objects:
                    for uid, obj in self._current_objects.items():
                        self._br.sendTransform(
                            obj.xyz_med, tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(), obj.name, "map"
                        )
            time.sleep(1./self.tf_publish_freq)


class InstructionListener:
    def __init__(self):
        self.instructions = []
        self.instructions_lock = threading.Lock()
        self.instruction_sub = rospy.Subscriber("/message", String, self.instructions_cb)

    def instructions_cb(self, msg):
        with self.instructions_lock:
            self.instructions.append(msg.data)

    def get_latest_human_side_instruction(self):
        with self.instructions_lock:
            for instruction in reversed(self.instructions):
                if "left" in instruction:
                    return "left"
                elif "right" in instruction:
                    return "right"
                else:
                    continue
            return None
