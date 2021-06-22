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
from sklearn.cluster import DBSCAN


from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionGoal
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Header, String


from rospy_message_converter import message_converter



base_vel_pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=1)


navclient = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
nav_client_initialized = False
while not nav_client_initialized:
    nav_client_initialized = navclient.wait_for_server(timeout=rospy.Duration(20.0))
    if nav_client_initialized:
        rospy.loginfo("Navigation Action Server Connected")
    else:
        rospy.logwarn("Unable to connect to Navigation Action Server")

arm = moveit_commander.MoveGroupCommander('arm')


gripper = moveit_commander.MoveGroupCommander("gripper")


head = moveit_commander.MoveGroupCommander("head")


base = moveit_commander.MoveGroupCommander("base")
base.allow_replanning(True)


tf_listener = tf.TransformListener()


def move_base_vel(vx, vy, vw):
    twist = Twist()
    twist.linear.x = vx
    twist.linear.y = vy
    twist.angular.z = math.radians(vw)
    base_vel_pub.publish(twist)


def move_base_actual_goal(goal):
    navclient.send_goal(goal)
    navclient.wait_for_result()
    state = navclient.get_state()
    return True if state == 3 else False


def get_current_time_sec():
    return rospy.Time.now().to_sec()


def quaternion_from_euler(roll, pitch, yaw):
    q = tf.transformations.quaternion_from_euler(
        math.radians(roll), math.radians(pitch), math.radians(yaw), 'rxyz'
    )
    return Quaternion(q[0], q[1], q[2], q[3])


def move_base_goal(x, y, theta):
    goal = MoveBaseGoal()

    goal.target_pose.header.frame_id = "map"

    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y

    goal.target_pose.pose.orientation = quaternion_from_euler(0, 0, theta)

    navclient.send_goal(goal)
    navclient.wait_for_result()
    state = navclient.get_state()

    return True if state == 3 else False


def get_diff_between(target_frame, source_frame):
    tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(0),rospy.Duration(4.0))
    transform = tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
    return transform[0][0], transform[0][1]


def move_arm_neutral():
    arm.set_named_target('neutral')
    return arm.go()


def move_arm_init():
    arm.set_named_target('go')
    return arm.go()


def move_hand(v):
    gripper.set_joint_value_target("hand_motor_joint", v)
    success = gripper.go()
    return success


def move_head_tilt(v):
    head.set_joint_value_target("head_tilt_joint", v)
    return head.go()


def get_object_dict():
    object_dict = {}
    paths = glob.glob("/opt/ros/melodic/share/tmc_wrs_gazebo_worlds/models/ycb*")
    for path in paths:
        file = os.path.basename(path)
        object_dict[file[8:]] = file

    return object_dict


def get_object_list():
    object_list = get_object_dict().values()
    object_list.sort()
    for i in range(len(object_list)):
        object_list[i] = object_list[i][8:]

    return object_list


def put_object(name, x, y, z):
    cmd = "rosrun gazebo_ros spawn_model -database " \
          + str(get_object_dict()[name]) \
          + " -sdf -model " + str(name) \
          + " -x " + str(y - 2.1) + \
          " -y " + str(-x + 1.2) \
          + " -z " + str(z)
    subprocess.call(cmd.split())


def delete_object(name):
    cmd = ['rosservice', 'call', 'gazebo/delete_model',
           '{model_name: ' + str(name) + '}']
    subprocess.call(cmd)


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


class ColorBasedObjectDetector:
    FLOOR_MIN_HUE, FLOOR_MAX_HUE = 14, 30

    def __init__(self, start_on_init=True):
        self._br = tf.TransformBroadcaster()
        self._cloud_sub = None

        self._current_objects = None
        self.lock = threading.Lock()

        if start_on_init:
            self.start()

    def start(self):
        if not self._cloud_sub:
            self._cloud_sub = rospy.Subscriber(
                "/hsrb/head_rgbd_sensor/depth_registered/rectified_points",
                PointCloud2, self._cloud_cb
            )

    def pause(self):
        if self._cloud_sub:
            self._cloud_sub.unregister()
            self._cloud_sub = None
            self._current_objects = None

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
            if uid in uid_to_pixels:
                uid_to_pixels[uid].append(PixelData(pixel, hue, x, y, z))
            else:
                uid_to_pixels[uid] = [PixelData(pixel, hue, x, y, z)]

        current_objects = {uid: SegmentedObject(uid, pixels, msg.header) for uid, pixels in uid_to_pixels.items()}

        for uid, obj in current_objects.items():
            self._br.sendTransform(
                obj.xyz_med, tf.transformations.quaternion_from_euler(0, 0, 0),
                rospy.Time(obj.header.stamp.secs, obj.header.stamp.nsecs),
                obj.name, obj.header.frame_id
            )

        with self.lock:
            self._current_objects = current_objects


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
