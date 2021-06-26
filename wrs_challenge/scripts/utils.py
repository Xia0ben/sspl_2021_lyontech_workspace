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
from shapely.geometry import Polygon, MultiPoint
from shapely import affinity


from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped, Twist, PointStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionGoal 
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Header, String


from rospy_message_converter import message_converter, json_message_converter



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

GROUND_OBJECTS_AREA = Polygon([(-0.22895914316177368, 1.5625046491622925), (-0.2523718774318695, 0.9900378584861755), (1.6986922025680542, 0.9900378584861755), (1.6908880472183228, 1.5182687044143677)])
LARGE_TABLE_OBJECTS_AREA = Polygon([(0.418794184923172, 2.0204780101776123), (0.4031856060028076, 1.6535788774490356), (1.6518666744232178, 1.6405681371688843), (1.6518666744232178, 1.9970588684082031)])
SMALL_TABLE_OBJECTS_AREA = Polygon([(-0.1925392746925354, 2.030886650085449), (0.17946362495422363, 2.030886650085449), (0.16125373542308807, 1.671793818473816), (-0.200343519449234, 1.6796001195907593)])
GROUND_OBJECTS_REDUCED_AREA = Polygon([(-0.21393196284770966, 0.8757089376449585), (-0.21393196284770966, 1.3273252248764038), (1.6966605186462402, 1.3163102865219116), (1.7131786346435547, 0.8702014088630676)])


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


def get_circumscribed_radius(polygon):
    center = list(polygon.centroid.coords)[0]
    points = list(polygon.exterior.coords)
    circumscribed_radius = 0.
    for point in points:
        circumscribed_radius = max(circumscribed_radius, euclidean_distance(center, point))
    return circumscribed_radius


def euclidean_distance(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def centroid(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length


class TimeWatchDogThread(threading.Thread):
    def run(self):
        start_time = rospy.Time.now()
        max_minutes = 14
        max_seconds = 50
        max_duration = max_minutes * 60 + max_seconds
        rospy.loginfo(
            "Movement started after {} minutes and {} seconds. Simulation will run for {} minutes and {} seconds.".format(
                str(start_time.secs // 60), str(start_time.secs % 60), str(max_minutes), str(max_seconds)
            )
        )
        while True:
            duration = (rospy.Time.now() - start_time).secs
            if duration > max_duration:
                rospy.loginfo("Interrupting simulation after {} minutes and {} seconds.".format(
                    str(duration // 60), str(duration % 60)
                ))
                time.sleep(1)
                os._exit(1)
            else:
                time.sleep(1)
            

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Robot(with_metaclass(Singleton)):
    JOINTS_FOR_SWIPING = [0.] + [math.radians(a) for a in [-146., 0., 53., 0., 0.]]
    GRASP_RADIUS = 0.7
    FULLY_CLOSED_GRIPPER_JOINTS = [
        0.025546200943420416,
        -0.11200161694114222,
        0.025001616941142224,
        0.028688636573056314,
        -0.11200161694114222
    ]
    FULLY_OPENED_GRIPPER_JOINTS = [
        0.006137660705964443,
        -1.0373166818629846,
        0.9503166818629847,
        0.0011030284262778522,
        -1.0373166818629846
    ]

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
        done = False
        while not done:
            try:
                self.base_vel_pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=1)
                done=True
            except Exception:
                done = False

    def init_arm(self):
        done = False
        while not done:
            try:
                self.arm = moveit_commander.MoveGroupCommander('arm')
                done=True
            except Exception:
                done = False

    def init_gripper(self):
        done = False
        while not done:
            try:
                self.gripper = moveit_commander.MoveGroupCommander("gripper")
                done=True
            except Exception:
                done = False

    def init_head(self):
        done = False
        while not done:
            try:
                self.head = moveit_commander.MoveGroupCommander("head")
                done=True
            except Exception:
                done = False

    def init_base(self):
        done = False
        while not done:
            try:
                self.base = moveit_commander.MoveGroupCommander("base")
                done=True
            except Exception:
                done = False

    def init_tf_listener(self):
        done = False
        while not done:
            try:
                self.tf_listener = tf.TransformListener()
                done=True
            except Exception:
                done = False

    def init_navclient(self):
        done = False
        while not done:
            try:
                self.navclient = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
                is_mb_up = self.is_move_base_up()
                while not is_mb_up:
                    rospy.logwarn("Unable to connect to Navigation Action Server, state: {}".format(is_mb_up))
                    is_mb_up = self.is_move_base_up()
                rospy.loginfo("Navigation Action Server Connected, state: {}".format(is_mb_up))
                done=True
            except Exception:
                done = False

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
    
    def is_hand_fully_closed(self):
        return all(np.isclose(self.gripper.get_current_joint_values(), self.FULLY_CLOSED_GRIPPER_JOINTS, atol=0.05))

    def move_head_tilt(self, v):
        self.head.set_joint_value_target("head_tilt_joint", v)
        return self.head.go()
    
    def move_arm_to_swiping_pose(self):
        self.arm.set_joint_value_target(self.JOINTS_FOR_SWIPING)
        return self.arm.go()
    
    def shake_wrist(self):
        arm_joints = self.arm.get_current_joint_values()
        arm_joints[2] += math.radians(10)
        self.arm.set_joint_value_target(arm_joints)
        self.arm.go()
        arm_joints[2] += math.radians(-10)
        self.arm.set_joint_value_target(arm_joints)
        self.arm.go()
        arm_joints[2] += math.radians(10)
        self.arm.set_joint_value_target(arm_joints)
        self.arm.go()
        arm_joints[2] += math.radians(-10)
        self.arm.set_joint_value_target(arm_joints)
        self.arm.go()

    
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
        
        self.top_pixels = [pixel for pixel in self.pixels if self.xyz_max[2] * 0.95 <= pixel.z <= self.xyz_max[2]]
        top_x = [pixel.x for pixel in self.pixels]
        top_y = [pixel.y for pixel in self.pixels]
        top_z = [pixel.z for pixel in self.pixels]
        self.xyz_avg_top = (np.average(top_x), np.average(top_y), np.average(top_z))

        self.pose = self.xyz_avg_top[0], self.xyz_avg_top[1], self.xyz_avg[2]

        self.hue_min = min(all_hues)
        self.hue_max = max(all_hues)
        self.hue_avg = np.average(all_hues)
        self.hue_med = np.median(all_hues)
        
        self.label = None
        
        self.name = "object_{}_with_hue_{}".format(self.uid, self.hue_med)
    
    def set_label(self, label):
        self.label = label
    
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
    
    @property
    def convex_footprint(self):
        return MultiPoint(self.bb_coords_2d).convex_hull
    
    @property
    def circumscribed_radius(self):
        convex_footprint = self.convex_footprint
        if isinstance(convex_footprint, Polygon):
            return get_circumscribed_radius(convex_footprint)
        else:
            return 0.00000000001

class Scene(with_metaclass(Singleton)):
    FLOOR_MIN_HUE, FLOOR_MAX_HUE = 14, 30
    
    def __init__(self, start_on_init=True, db_scan_eps=4, db_scan_min=10):
        self.db_scan_eps = db_scan_eps
        self.db_scan_min = db_scan_min
        
        self.use_labels = False
        
        self._br = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()
        self.obj_detector = ObjectDetection()
        
        self._cloud_sub = None
        
        self._current_objects = None
        self.lock = threading.Lock()
        
        self.tf_publish_freq = 10.
        self.tf_publishing_thread = threading.Thread(target=self.publish_current_objects_tfs)
        self.tf_publishing_thread.start()
        
        self.objects_region = None
        
        if start_on_init:
            self.start()
        
    def start(self):
        with self.lock:
            self._current_objects = None
        if not self._cloud_sub:
            self._cloud_sub = rospy.Subscriber(
                "/hsrb/head_rgbd_sensor/depth_registered/rectified_points",
                PointCloud2, self._cloud_cb
            )
    
    def pause(self):
        if self._cloud_sub:
            self._cloud_sub.unregister()
            self._cloud_sub = None
    
    def wait_for_one_detection(self, timeout=20., sleep_duration=0.0001, use_labels=False):       
        start_time = time.time()
        self.use_labels = use_labels
        self.start()
        current_objects = {}
        now = time.time()
        while now - start_time < timeout:
            if self._current_objects is not None:
                with self.lock:
                    current_objects = self._current_objects
                    self.pause()
                    return current_objects
            else:
                time.sleep(sleep_duration)
            now = time.time()
        self.pause()
        self.use_labels = False
        return current_objects

    def _cloud_cb(self, msg, debug=True):
        rospy.loginfo("Cloud Callback called.")
        
        points = ros_numpy.numpify(msg)

        image = points['rgb'].view((np.uint8, 4))[..., [2, 1, 0]]

        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
        h_image = hsv_image[..., 0]

        floor_region = (h_image > self.FLOOR_MIN_HUE) & (h_image < self.FLOOR_MAX_HUE)
        wall_and_robot_region = h_image == 0
        objects_region = wall_and_robot_region | floor_region
        
        if debug:
            self.objects_region = objects_region
                
        object_pixels = zip(*np.where(objects_region==False))
        
        if not object_pixels:
            with self.lock:
                self._current_objects = {}
            return

        db_scan_result = DBSCAN(eps=self.db_scan_eps, min_samples=self.db_scan_min).fit(object_pixels)

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
        
        if self.use_labels:
            # list(tuple(minx, miny, maxx, maxy, cx, cy, label, confidence))
            detector_output = self.obj_detector.detect(image)
#             print("detector_output: {}".format(detector_output))
            objs_pixels_centroids = {
                uid: centroid(np.array([pixel.pixel for pixel in obj.pixels]))
                for uid, obj in new_objects.items()
            }
#             print("objs_pixels_centroids: {}".format(objs_pixels_centroids))
            for uid, c in objs_pixels_centroids.items():
                label_by_distance = []
                for (minx, miny, maxx, maxy, cx, cy, label, confidence) in detector_output:
                    if minx <= c[0] <= maxx and miny <= c[1] <= maxy:
                        label_by_distance.append((label, euclidean_distance(c, (cx, cy))))
                label_by_distance = sorted(label_by_distance, key=lambda tup: tup[1])
                if label_by_distance:
                    new_objects[uid].set_label(label_by_distance[0][0])
            
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
                            obj.pose, tf.transformations.quaternion_from_euler(0, 0, 0),
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
        
        
class ObjectDetection(object):
    def __init__(self):
        # Get real-time video stream through opencv
        LABELS_FILE = 'ycb_tinyyolo/ycb_simu.names'
        CONFIG_FILE = 'ycb_tinyyolo/yolov3-tiny-ycb_simu_test.cfg'
        WEIGHTS_FILE = 'ycb_tinyyolo/yolov3-tiny-ycb_simu_best_004.weights'
        self.CONFIDENCE_THRESHOLD = 0.3

        self.H = None
        self.W = None

        self.LABELS = open(LABELS_FILE).read().strip().split('\n')
        np.random.seed(4)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS),
                3), dtype='uint8')

        self.net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

        self.ln = self.net.getLayerNames()
        self.ln = [
            self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()
        ]

    def detect(self, image):
        data = []
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
        self.net.setInput(blob)
        if self.W is None or self.H is None:
            (self.H, self.W) = image.shape[:2]

        layerOutputs = self.net.forward(self.ln)

            # initialize our lists of detected bounding boxes, confidences, and
            # class IDs, respectively

        boxes = []
        confidences = []
        classIDs = []

            # loop over each of the layer outputs

        for output in layerOutputs:

                # loop over each of the detections

            for detection in output:

                    # extract the class ID and confidence (i.e., probability) of
                    # the current object detection

                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability

                if confidence > self.CONFIDENCE_THRESHOLD:

                        # scale the bounding box coordinates back relative to the
                        # size of the image, keeping in mind that YOLO actually
                        # returns the center (x, y)-coordinates of the bounding
                        # box followed by the boxes' width and height

                    box = detection[0:4] * np.array([self.W, self.H,
                            self.W, self.H])
                    (centerX, centerY, width, height) = box.astype('int'
                            )

                        # use the center (x, y)-coordinates to derive the top and
                        # and left corner of the bounding box

                    x = int(centerX - width / 2)
                    y = int(centerY - height / 2)

                        # update our list of bounding box coordinates, confidences,
                        # and class IDs

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes

        idxs = cv2.dnn.NMSBoxes(boxes, confidences,
                                self.CONFIDENCE_THRESHOLD,
                                self.CONFIDENCE_THRESHOLD)

            # ensure at least one detection exists

        if len(idxs) > 0:

                # loop over the indexes we are keeping

            for i in idxs.flatten():

                    # extract the bounding box coordinates

                (miny, minx) = (boxes[i][0], boxes[i][1])
                (maxy, maxx) = (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3])
                cx, cy = centroid(np.array([(minx, miny), (maxx, maxy)]))
                
                tuple_i = (minx, miny, maxx, maxy, cx, cy, self.LABELS[classIDs[i]], confidences[i])
                data.append(tuple_i)

            # print(data)

        return data

class MessageParser():
    def __init__(self):
        self.lock = threading.Lock()

        self.person = "pending"
        self.object = "pending"
        self.object_num = -1
        self.object_darknet = "pending"

        self.load_dictionnaries()
        
        rospy.Subscriber("/message", String, self.message)

    def get_person(self):
        with self.lock:
            return self.person
    
    def get_object(self):
        with self.lock:
            return self.object
        
    def get_object_num(self):
        with self.lock:
            return self.object_num
        
    def get_object_darknet(self):
        with self.lock:
            return self.object_darknet
        
    # /message Topic callback
    def message(self, msg):
        with self.lock:
            if("left" in msg.data.lower()):
                self.person = "left"
            elif("right" in msg.data.lower()):
                self.person = "right"
            else:
                self.person = "undefined"

            num, name = self.match_object(msg.data)

            if( num == 0 ):
                self.object = "undefined"
                self.object_darknet = "undefined"
            else:
                self.object = name

            self.object_num = num

            if( num > 0 ):
                self.object_darknet = self.ycb_num_to_darknet_label(num)
                
    # check if there is an object name of the ycb dataset is in the /message request
    def match_object(self, smsg):

        match = False
        
        for num, names in self.ycb_number_to_ycb_names.items():
            if( len(names) > 0 ):
                for name in names:
                    if( name.lower() in smsg.lower() ):
                        return num, name
                    if( name.replace('_',' ').lower() in smsg.lower() ):
                        return num, name                        
        
        return 0, "undefined"
    
    def ycb_num_to_darknet_label(self, num):
        # list out keys and values separately
        key_list = list(self.darknet_label_to_ycb_number.keys())
        val_list = list(self.darknet_label_to_ycb_number.values())
        
        # print key with val num
        try:
            position = val_list.index(num)
            return key_list[position] 
        except:
            return "undefined"
        
    def get_deposit(self, darknet_label):
        ycb_num = self.darknet_label_to_ycb_number[darknet_label]

        if ycb_num in self.ycbNumObj_deposit.keys():
            deposit = self.ycbNumObj_deposit[ycb_num]["Deposit"]
        else:
            deposit = []

        return deposit
        
    # Load conversion dictionnaries
    def load_dictionnaries(self):

        self.ycb_number_to_ycb_names =    \
            {
                1:["chips_can"],
                2:["master_chef_can", "coffee"],
                3:["cracker_box", "Cheez-it"],
                4:["sugar_box"],
                5:["tomato_soup_can"],
                6:["mustard_bottle"],
                7:["tuna_fish_can"],
                8:["pudding_box"],
                9:["gelatin_box"],
                10:["potted_meat_can"],
                11:["banana"],
                12:["strawberry"],
                13:["apple"],
                14:["lemon"],
                15:["peach"],
                16:["pear"],
                17:["orange"],
                18:["plum"],
                19:["pitcher_base"],
                20:[],
                21:["bleach_cleanser"],
                22:["windex_bottle"],
                23:["wine_glass"],
                24:["bowl"],
                25:["mug"],
                26:["sponge"],
                27:["skillet"],
                28:["skillet_lid"],
                29:["plate"],
                30:["fork"],
                31:["spoon"],
                32:["knife"],
                33:["spatula"],
                34:[],
                35:["power_drill"],
                36:["wood_block"],
                37:["scissors"],
                38:["padlock"],
                39:["key"],
                40:["large_marker"],
                41:["small_marker"],
                42:["adjustable_wrench"],
                43:["phillips_screwdriver"],
                44:["flat_screwdriver"],
                45:[],
                46:["plastic_bolt"],
                47:["plastic_nut"],
                48:["hammer"],
                49:["small_clamp"],
                50:["medium_clamp"],
                51:["large_clamp"],
                52:["extra_large_clamp"],
                53:["mini_soccer_ball"],
                54:["softball"],
                55:["baseball"],
                56:["tennis_ball"],
                57:["racquetball"],
                58:["golf_ball"],
                59:["chain"],
                60:[],
                61:["foam_brick"],
                62:["dice"],
                63:["marbles", "a_marbles", "b_marbles", "c_marbles", "d_marbles", "e_marbles", "f_marbles"],
                62:[],
                63:[],
                64:[],
                65:["cups", "a_cups", "b_cups", "c_cups", "d_cups", "e_cups","f_cups","g_cups","h_cups","i_cups","j_cups"],
                66:[],
                67:[],
                68:[],
                69:[],
                70:["colored_wood_blocks", "a_colored_wood_blocks", "b_colored_wood_blocks"],
                71:["nine_hole_peg_test"],
                72:["toy_airplane", "a_toy_airplane", "b_toy_airplane", "c_toy_airplane", "d_toy_airplane", "e_toy_airplane", "f_toy_airplane", "g_toy_airplane", "h_toy_airplane", "i_toy_airplane","j_toy_airplane", "k_toy_airplane"],
                73:["a_lego_duplo","b_lego_duplo", "c_lego_duplo", "d_lego_duplo", "e_lego_duplo", "f_lego_duplo", "g_lego_duplo", "h_lego_duplo", "i_lego_duplo", "j_lego_duplo", "k_lego_duplo", "l_lego_duplo", "m_lego_duplo"],
                74:[],
                75:[],
                76:["timer"],
                77:["rubiks_cube"]
            }


        self.darknet_label_to_ycb_number = \
            {
                "cracker"               : 3 , 
                "sugar"                 : 4 ,
                "pudding"               : 8 ,
                "gelatin"               : 9 ,
                "pottedmeat"            : 10,
                "coffee"                : 2 ,
                "tuna"                  : 7 ,
                "chips"                 : 1 ,
                "mustard"               : 6 ,
                "tomatosoup"            : 5 ,
                "banana"                : 11,
                "strawberry"            : 12,
                "apple"                 : 13,
                "lemon"                 : 14,
                "peach"                 : 15,
                "pear"                  : 16,
                "orange"                : 17,
                "plum"                  : 18,
                "windex"                : 22,
                "bleach"                : 21,
                "pitcher"               : 19,
                "plate"                 : 29,
                "bowl"                  : 24,
                "fork"                  : 30,
                "spoon"                 : 26,
                "spatula"               : 33,
                "wineglass"             : 23,
                "cup"                   : 65,
                "largemarker"           : 40,
                "smallmarker"           : 41,
                "padlocks"              : 38,
                "bolt"                  : 46,
                "nut"                   : 47,
                "clamp"                 : 49,
                "soccerball"            : 53,
                "baseball"              : 55,
                "tennisball"            : 56,
                "golfball"              : 58,
                "foambrick"             : 61,
                "dice"                  : 62,
                "rope"                  : -1,
                "chain"                 : 59,
                "rubikscube"            : 77,
                "coloredwoodblock"      : 70,
                "peghole"               : 71,
                "timer"                 : 76,
                "airplane"              : 72,
                "tshirt"                : -1,
                "magazine"              : -1,
                "creditcard"            : -1,
                "legoduplo"             : 73,
                "sponge"                : 26,
                "coloredwoodblockpot"   : 70,
                "softball"              : 54,
                "racquetball"           : 57,
                "marbles"               : 63,
                "mug"                   : 25
            }



        self.ycbNumObj_deposit =    \
            {
                1:{"Category":"food", "Deposit": ["Tray_A", "Tray_B"], "Place":["Long_Table_A"]},
                2:{"Category":"food", "Deposit": ["Tray_A", "Tray_B"], "Place":["Long_Table_A"]},
                3:{"Category":"food", "Deposit": ["Tray_A", "Tray_B"], "Place":["Long_Table_A"]},
                4:{"Category":"food", "Deposit": ["Tray_A", "Tray_B"], "Place":["Long_Table_A"]},
                5:{"Category":"food", "Deposit": ["Tray_A", "Tray_B"], "Place":["Long_Table_A"]},
                6:{"Category":"food", "Deposit": ["Tray_A", "Tray_B"], "Place":["Long_Table_A"]},
                7:{"Category":"food", "Deposit": ["Tray_A", "Tray_B"], "Place":["Long_Table_A"]},
                8:{"Category":"food", "Deposit": ["Tray_A", "Tray_B"], "Place":["Long_Table_A"]},
                9:{"Category":"food", "Deposit": ["Tray_A", "Tray_B"], "Place":["Long_Table_A"]},
                10:{"Category":"food", "Deposit": ["Tray_A", "Tray_B"], "Place":["Long_Table_A"]},
                11:{"Category":"food", "Deposit": ["Tray_A", "Tray_B"], "Place":["Long_Table_A"]},
                12:{"Category":"food", "Deposit": ["Tray_A", "Tray_B"], "Place":["Long_Table_A"]},
                13:{"Category":"food", "Deposit": ["Tray_A", "Tray_B"], "Place":["Long_Table_A"]},
                14:{"Category":"food", "Deposit": ["Tray_A", "Tray_B"], "Place":["Long_Table_A"]},
                15:{"Category":"food", "Deposit": ["Tray_A", "Tray_B"], "Place":["Long_Table_A"]},
                16:{"Category":"food", "Deposit": ["Tray_A", "Tray_B"], "Place":["Long_Table_A"]},
                17:{"Category":"food", "Deposit": ["Tray_A", "Tray_B"], "Place":["Long_Table_A"]},
                18:{"Category":"food", "Deposit": ["Tray_A", "Tray_B"], "Place":["Long_Table_A"]},

                19:{"Category":"kitchenitem", "Deposit": ["Container_A"], "Place":["Long_Table_A"]},
                20:{"Category":"kitchenitem", "Deposit": ["Container_A"], "Place":["Long_Table_A"]},
                21:{"Category":"kitchenitem", "Deposit": ["Container_A"], "Place":["Long_Table_A"]},
                22:{"Category":"kitchenitem", "Deposit": ["Container_A"], "Place":["Long_Table_A"]},
                23:{"Category":"kitchenitem", "Deposit": ["Container_A"], "Place":["Long_Table_A"]},
                24:{"Category":"kitchenitem", "Deposit": ["Container_A"], "Place":["Long_Table_A"]},
                25:{"Category":"kitchenitem", "Deposit": ["Container_A"], "Place":["Long_Table_A"]},
                26:{"Category":"kitchenitem", "Deposit": ["Container_A"], "Place":["Long_Table_A"]},
                27:{"Category":"kitchenitem", "Deposit": ["Container_A"], "Place":["Long_Table_A"]},
                28:{"Category":"kitchenitem", "Deposit": ["Container_A"], "Place":["Long_Table_A"]},
                29:{"Category":"kitchenitem", "Deposit": ["Container_A"], "Place":["Long_Table_A"]},
                30:{"Category":"kitchenitem", "Deposit": ["Container_A"], "Place":["Long_Table_A"]},
                31:{"Category":"kitchenitem", "Deposit": ["Container_A"], "Place":["Long_Table_A"]},
                32:{"Category":"kitchenitem", "Deposit": ["Container_A"], "Place":["Long_Table_A"]},
                33:{"Category":"kitchenitem", "Deposit": ["Container_A"], "Place":["Long_Table_A"]},


                38:{"Category":"tool", "Deposit": ["Drawer_top", "Drawer_bottom"], "Place":["Drawer"]},


                40:{"Category":"tool", "Deposit": ["Drawer_top", "Drawer_bottom"], "Place":["Drawer"]},
                41:{"Category":"tool", "Deposit": ["Drawer_top", "Drawer_bottom"], "Place":["Drawer"]},


                46:{"Category":"tool", "Deposit": ["Drawer_top", "Drawer_bottom"], "Place":["Drawer"]},
                47:{"Category":"tool", "Deposit": ["Drawer_top", "Drawer_bottom"], "Place":["Drawer"]},
                48:{"Category":"tool", "Deposit": ["Drawer_top", "Drawer_bottom"], "Place":["Drawer"]},
                49:{"Category":"tool", "Deposit": ["Drawer_top", "Drawer_bottom"], "Place":["Drawer"]},
                50:{"Category":"tool", "Deposit": ["Drawer_top", "Drawer_bottom"], "Place":["Drawer"]},
                51:{"Category":"tool", "Deposit": ["Drawer_top", "Drawer_bottom"], "Place":["Drawer"]},
                52:{"Category":"tool", "Deposit": ["Drawer_top", "Drawer_bottom"], "Place":["Drawer"]},
                
                53:{"Category":"shapeitem", "Deposit": ["Drawer_left"], "Place":["Drawer"]},
                54:{"Category":"shapeitem", "Deposit": ["Drawer_left"], "Place":["Drawer"]},
                55:{"Category":"shapeitem", "Deposit": ["Drawer_left"], "Place":["Drawer"]},
                56:{"Category":"shapeitem", "Deposit": ["Drawer_left"], "Place":["Drawer"]},
                57:{"Category":"shapeitem", "Deposit": ["Drawer_left"], "Place":["Drawer"]},
                58:{"Category":"shapeitem", "Deposit": ["Drawer_left"], "Place":["Drawer"]},
                59:{"Category":"shapeitem", "Deposit": ["Drawer_left"], "Place":["Drawer"]},
                60:{"Category":"shapeitem", "Deposit": ["Drawer_left"], "Place":["Drawer"]},
                61:{"Category":"shapeitem", "Deposit": ["Drawer_left"], "Place":["Drawer"]},
                62:{"Category":"shapeitem", "Deposit": ["Drawer_left"], "Place":["Drawer"]},
                63:{"Category":"shapeitem", "Deposit": ["Drawer_left"], "Place":["Drawer"]},
                62:{"Category":"shapeitem", "Deposit": ["Drawer_left"], "Place":["Drawer"]},
                63:{"Category":"shapeitem", "Deposit": ["Drawer_left"], "Place":["Drawer"]},
                64:{"Category":"shapeitem", "Deposit": ["Drawer_left"], "Place":["Drawer"]},
                65:{"Category":"shapeitem", "Deposit": ["Drawer_left"], "Place":["Drawer"]},


                70:{"Category":"taskitem", "Deposit": ["Bin_A"], "Place":["Bin_A"]},
                71:{"Category":"taskitem", "Deposit": ["Bin_A"], "Place":["Bin_A"]},
                72:{"Category":"taskitem", "Deposit": ["Bin_A"], "Place":["Bin_A"]},
                73:{"Category":"taskitem", "Deposit": ["Bin_A"], "Place":["Bin_A"]},


                76:{"Category":"taskitem", "Deposit": ["Bin_A"], "Place":["Bin_A"]},

                77:{"Category":"taskitem", "Deposit": ["Bin_A"], "Place":["Bin_A"]}
            }   
        

##################################################
#                     LAYOUT
##################################################

##########################################
#  ___________  ___________    ____  ____      __   ____
# |           ||           |  |    ||    |    |CB| | CA |
# |           ||           |  | TB || TA |         |____|
# |   BIN_B   ||   BIN_A   |  |____||____|
# |           ||           |
# |___________||___________|
#
#       B1           B2         P1     P2            B3


##################################################
#                     BOXES
##################################################


# ******************** BOX_6 ********************
# | 1 2 3 |
# |       |
# | 4 5 6 |

BIN_B_6 = [(2.9520626068115234, -0.6928567290306091), (2.86234712600708, -0.6945728659629822), (2.74869441986084, -0.691084623336792), (2.9655299186706543, -0.5568759441375732), (2.86177659034729, -0.5359254479408264), (2.7385966777801514, -0.5274717807769775)]
BIN_A_6 = [(2.4541680812835693, -0.6813899278640747), (2.3308026790618896, -0.6841501593589783), (2.205941915512085, -0.6804757118225098), (2.460055351257324, -0.5196448564529419), (2.3544352054595947, -0.5146874785423279), (2.248734474182129, -0.5145358443260193)]
CONTAINER_A_6 = [(1.12795090675354, -0.640928328037262), (1.0497441291809082, -0.6381940245628357), (0.9671953916549683, -0.6375424265861511), (1.1300628185272217, -0.5569236278533936), (1.047502040863037, -0.5569902062416077), (0.9607541561126709, -0.5498043298721313)]

# ******************** BOX_5 ********************
# | 1   4 |
# |   3   |
# | 5   6 |

BIN_B_5 = [(2.9520626068115234, -0.6928567290306091), (2.86234712600708, -0.6945728659629822), (2.861199140548706, -0.6069657802581787), (2.86177659034729, -0.5359254479408264), (2.7385966777801514, -0.5274717807769775)]
BIN_A_5 = [(2.4541680812835693, -0.6813899278640747), (2.3308026790618896, -0.6841501593589783), (2.3573896884918213, -0.5887709856033325), (2.3544352054595947, -0.5146874785423279), (2.248734474182129, -0.5145358443260193)]
CONTAINER_A_5 = [(1.12795090675354, -0.640928328037262), (1.0497441291809082, -0.6381940245628357), (1.0439521074295044, -0.5978738069534302), (1.047502040863037, -0.5569902062416077), (0.9607541561126709, -0.5498043298721313)]


# ******************** BOX_4 ********************
# | 1   2 |
# |       |
# | 3   4 |

BIN_B_4 =  [(2.942748546600342, -0.6750758290290833), (2.7617690563201904, -0.6752774119377136), (2.9462902545928955, -0.5581591129302979), (2.7847368717193604, -0.5458639860153198)]
BIN_A_4 = [(2.441438913345337, -0.6763715147972107), (2.2364697456359863, -0.6745726466178894), (2.436012029647827, -0.5208483338356018), (2.2439043521881104, -0.516058087348938)]
CONTAINER_A_4 = [(1.1122328042984009, -0.6363577842712402), (0.9874104857444763, -0.6306950449943542), (1.1079329252243042, -0.5493735671043396), (0.9701082706451416, -0.5485230088233948)]

# ******************** BOX_3 ********************
# |   1   |
# |       |
# | 2   3 |

BIN_B_3 = [(2.8604800701141357, -0.710565984249115), (2.9494919776916504, -0.5582122802734375), (2.7670180797576904, -0.5519794821739197)]
BIN_A_3 = [(2.3417177200317383, -0.701957643032074), (2.4376132488250732, -0.5208748579025269), (2.2358195781707764, -0.5207310914993286)]
CONTAINER_A_3 = [(1.0482254028320312, -0.6431968212127686), (1.1200505495071411, -0.5546025037765503), (0.9672135710716248, -0.5499115586280823)]

# ******************** BOX_4 ********************
# | 1     |
# |       |
# |     2 |

BIN_B_2 = [(2.9503817558288574, -0.6976361274719238), (2.7622148990631104, -0.551899790763855)]
BIN_A_2 = [(2.4237470626831055, -0.6808851957321167), (2.2342448234558105, -0.5191025733947754)]
CONTAINER_A_2 = [(1.1150918006896973, -0.6371235251426697), (0.9756474494934082, -0.5608258247375488)]

# ******************** BOX_4 ********************
# |       |
# |   1   |
# |       |

BIN_B_1 = [(2.8525211811065674, -0.6110848784446716)]
BIN_A_1 = [(2.3226048946380615, -0.5990867614746094)]
CONTAINER_A_1 = [(1.0453635454177856, -0.599333643913269)]



##################################################
#                     PLATE
##################################################

# ******************** PLATE_6 ********************
# | 1 2 |
# | 3 4 |
# | 5 6 |

TRAY_B_6 = [(1.959380030632019, -0.7126528024673462), (1.8296968936920166, -0.7000574469566345), (1.9529002904891968, -0.6139144897460938), (1.823005199432373, -0.6140797138214111), (1.9639467000961304, -0.5073444843292236), (1.819152593612671, -0.49681925773620605)]
TRAY_A_6 = [(1.6729234457015991, -0.7125407457351685), (1.5198979377746582, -0.7088410258293152), (1.6699410676956177, -0.6127001047134399), (1.5249933004379272, -0.6114552617073059), (1.6811803579330444, -0.49452975392341614), (1.5200772285461426, -0.48837536573410034)]

# ******************** PLATE_5-1 ********************
# Same layout as box for the rest

TRAY_B_5 = [(1.959380030632019, -0.7126528024673462), (1.8296968936920166, -0.7000574469566345), (1.8904836177825928, -0.6012750864028931), (1.9639467000961304, -0.5073444843292236), (1.819152593612671, -0.49681925773620605)]
TRAY_A_5 = [(1.6729234457015991, -0.7125407457351685), (1.5198979377746582, -0.7088410258293152), (1.6039307117462158, -0.6069632768630981), (1.5249933004379272, -0.6114552617073059), (1.6811803579330444, -0.49452975392341614), (1.5200772285461426, -0.48837536573410034)]

TRAY_B_4 = [(1.9678038358688354, -0.6942266821861267), (1.83456552028656, -0.6862139701843262), (1.9685651063919067, -0.5085815191268921), (1.8237324953079224, -0.5003763437271118)]
TRAY_A_4 = [(1.6626042127609253, -0.7054073214530945), (1.535163164138794, -0.6974908709526062), (1.6669976711273193, -0.5105394721031189), (1.529083251953125, -0.5047699213027954)]

TRAY_B_3 = [(1.9686747789382935, -0.7116466760635376), (1.8904836177825928, -0.6012750864028931), (1.819267988204956, -0.489859014749527)]
TRAY_A_3 = [(1.6742756366729736, -0.7009596228599548), (1.6039307117462158, -0.6069632768630981), (1.529179334640503, -0.49896958470344543)]

TRAY_B_2 = [(1.9652925729751587, -0.7057886719703674), (1.8387089967727661, -0.5064266920089722)]
TRAY_A_2 = [(1.6477432250976562, -0.6923967599868774), (1.541702151298523, -0.5131017565727234)]

TRAY_B_1 = (1.8904836177825928, -0.6012750864028931)
TRAY_A_1 = (1.6039307117462158, -0.6069632768630981)


HEIGHT_ABOVE_BINS = 0.5
HEIGHT_ABOVE_TRAYS = 0.53
HEIGHT_ABOVE_CONTAINER_A = 0.62

in_front_large_table_ground_objects_goal_str = '{"header": {"stamp": {"secs": 824, "nsecs": 610000000}, "frame_id": "", "seq": 0}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 824, "nsecs": 598000000}, "frame_id": "map", "seq": 0}, "pose": {"position": {"y": 0.24864110350608826, "x": 1.026589274406433, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.70455870740194, "w": 0.7096457058449008}}}}}'
IN_FRONT_LARGE_TABLE_GROUND_OBJECTS_GOAL = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', in_front_large_table_ground_objects_goal_str).goal

in_front_deposit_table_goal_str = '{"header": {"stamp": {"secs": 972, "nsecs": 594000000}, "frame_id": "", "seq": 4}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 972, "nsecs": 594000000}, "frame_id": "map", "seq": 0}, "pose": {"position": {"y": 0.4221145510673523, "x": 1.5025765895843506, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": -0.7071067966408575, "w": 0.7071067657322372}}}}}'
IN_FRONT_DEPOSIT_TABLE_GOAL = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', in_front_deposit_table_goal_str).goal

in_front_bins_goal_str = '{"header": {"stamp": {"secs": 269, "nsecs": 985000000}, "frame_id": "", "seq": 1}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 269, "nsecs": 979000000}, "frame_id": "map", "seq": 1}, "pose": {"position": {"y": 0.3244279623031616, "x": 2.6250836849212646, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": -0.7071067966408575, "w": 0.7071067657322372}}}}}'
IN_FRONT_BINS_GOAL = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', in_front_bins_goal_str).goal