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


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Robot(with_metaclass(Singleton)):
    JOINTS_FOR_SWIPING = [0.] + [math.radians(a) for a in [-146., 0., 53., 0., 0.]]
    GRASP_RADIUS = 0.7

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

        self.label = None

        self.name = "object_with_hue_{}".format(self.hue_med)

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

class Scene(with_metaclass(Singleton)):
    FLOOR_MIN_HUE, FLOOR_MAX_HUE = 14, 30

    def __init__(self, start_on_init=True):
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

    def wait_for_one_detection(self, timeout=10., sleep_duration=0.0001, use_labels=False):
        start_time = time.time()
        self.use_labels = use_labels
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
        self.use_labels = False
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


class ObjectDetection(object):
    def __init__(self):
        # Get real-time video stream through opencv
        LABELS_FILE = '/workspace/src/wrs_challenge/scripts/ycb_tinyyolo/ycb_simu.names'
        CONFIG_FILE = '/workspace/src/wrs_challenge/scripts/ycb_tinyyolo/yolov3-tiny-ycb_simu_test.cfg'
        WEIGHTS_FILE = '/workspace/src/wrs_challenge/scripts/ycb_tinyyolo/yolov3-tiny-ycb_simu_best_004.weights'
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

    # Load YCB label dictionnary
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
