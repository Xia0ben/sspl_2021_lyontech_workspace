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
import traceback


from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionGoal
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Header, String


from rospy_message_converter import message_converter


def get_current_time_sec():
    return rospy.Time.now().to_sec()


def quaternion_from_euler(roll, pitch, yaw):
    q = tf.transformations.quaternion_from_euler(
        math.radians(roll), math.radians(pitch), math.radians(yaw), 'rxyz'
    )
    return Quaternion(q[0], q[1], q[2], q[3])


class Robot:
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
        success = self.gripper.go()
        return success

    def move_head_tilt(self, v):
        self.head.set_joint_value_target("head_tilt_joint", v)
        return self.head.go()



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
