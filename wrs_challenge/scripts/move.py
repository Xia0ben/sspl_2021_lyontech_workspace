#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import matplotlib.pyplot as plt


# In[3]:


import math

import os

import tf

import sys

import threading

import rospy
rospy.init_node("go_and_get_it_01")


# Wait for Gazebo to actually properly start...
import time
while rospy.Time.now() == rospy.Time():
    rospy.loginfo("Simulation paused/stalled")
    time.sleep(0.1)
rospy.loginfo("Simulation started")

from rospy_message_converter import json_message_converter


from geometry_msgs.msg import Pose, PointStamped

from shapely.geometry import MultiPoint, Polygon, Point


import utils

robot = utils.Robot()
scene = utils.Scene(start_on_init=False)
message_parser = utils.MessageParser()

rospy.loginfo("Imports done, robot initialized.")


# In[4]:


utils.NavGoalToJsonFileSaver("saved_msg.json")


# In[5]:


with open("saved_msg.json") as f:
    print(f.read())


# In[ ]:


# in_front_drawers_goal_str = '{"header": {"stamp": {"secs": 69, "nsecs": 237000000}, "frame_id": "", "seq": 2}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 69, "nsecs": 228000000}, "frame_id": "map", "seq": 2}, "pose": {"position": {"y": 0.5999923944473267, "x": 0.24766463041305542, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": -0.7189345607475961, "w": 0.6950777635363263}}}}}'
# in_front_drawers_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', in_front_drawers_goal_str).goal
# robot.move_base_actual_goal(in_front_drawers_goal)


# In[ ]:


# robot.open_hand()
# READY_FOR_BOTTOM_DRAWERS_ARM_JOINTS = [.25] + [math.radians(a) for a in [-150., 0., 60., 0., 0.]]
# robot.arm.set_joint_value_target(READY_FOR_BOTTOM_DRAWERS_ARM_JOINTS)
# robot.arm.go()


# In[ ]:


# GRASP_RIGHT_BOTTOM_RIGHT_DRAWER_BASE_JOINTS = [0.14590200293468028, 0.09235241684400744, -1.4923558765834182]
# robot.base.set_joint_value_target(GRASP_RIGHT_BOTTOM_RIGHT_DRAWER_BASE_JOINTS)
# robot.base.go()
# robot.close_hand()


# In[ ]:


# PULL_RIGHT_BOTTOM_RIGHT_DRAWER_BASE_JOINTS = [0.448629245701104, 0.08066860734930102, -1.511430367700255]
# robot.base.set_joint_value_target(PULL_RIGHT_BOTTOM_RIGHT_DRAWER_BASE_JOINTS)
# robot.base.go()
# robot.open_hand()


# In[ ]:


def get_chosen_object(cur_objects, previous_convex_footprints, pose_z_min, pose_z_max, xy_polygon):
    chosen_object = None
    # Choose closest object that fits in robot's hand by default otherwise
    uid_by_distance = []
    for uid, obj in cur_objects.items():
        if obj.circumscribed_radius <= robot.GRASP_RADIUS:
            intersects = False
            convex_footprint = obj.convex_footprint
            point = Point([obj.pose[0], obj.pose[1]])
            if pose_z_min <= obj.pose[2] <= pose_z_max and point.intersects(xy_polygon):
                for prev_cv_ft in previous_convex_footprints:
                    if prev_cv_ft.intersects(convex_footprint):
                        intersects = True
                        break
                if not intersects:
                    x, _= robot.get_diff_between("base_link", obj.name)
                    uid_by_distance.append((uid, x))

    uid_by_distance = sorted(uid_by_distance, key=lambda tup: tup[1])
    if uid_by_distance:
        chosen_object = cur_objects[uid_by_distance[0][0]]

    return chosen_object, uid_by_distance


# In[ ]:


def pick_object_away(obj, joints_for_hovering, lowest_arm_height):
    # Move head to prevent arm movement failures
    robot.move_head_tilt(0.)

    # Save joints for initial pose
    joints_for_going_back_to_init_pose = robot.base.get_current_joint_values()

    # Compute angle for robot base to face arm parallel direction between base_link and object
    o_x, o_y = robot.get_diff_between("base_link", obj.name)
    yaw = math.pi/2. - math.atan2(o_x, o_y)

    joints_for_facing_object = robot.base.get_current_joint_values()
    joints_for_facing_object[2] += yaw

    robot.base.set_joint_value_target(joints_for_facing_object)
    robot.base.go()

    # Set to picking pose
    robot.arm.set_joint_value_target(joints_for_hovering)
    is_success = robot.arm.go()
    robot.open_hand()

    # Compute translation for robot base to actually face the object
    a_x, a_y = robot.get_diff_between("base_link", "arm_flex_link")

    robot.tf_listener.waitForTransform("/base_link", "/odom", rospy.Time(0),rospy.Duration(4.0))
    point=PointStamped()
    point.header.frame_id = "base_link"
    point.header.stamp =rospy.Time(0)
    point.point.y= -a_y
    p=robot.tf_listener.transformPoint("odom", point)

    joints_for_going_to_object = robot.base.get_current_joint_values()
    joints_for_going_to_object[0] = p.point.y
    joints_for_going_to_object[1] = p.point.x

    robot.base.set_joint_value_target(joints_for_going_to_object)
    robot.base.go()

    # Compute translation for robot base to get the object and get it
    oo_x, oo_y = robot.get_diff_between("odom", obj.name)
    ho_x, ho_y = robot.get_diff_between("odom", "hand_palm_link")

    joints_for_catching_to_object = robot.base.get_current_joint_values()
    joints_for_catching_to_object[0] += oo_y - ho_y
    joints_for_catching_to_object[1] += oo_x - ho_x

    robot.base.set_joint_value_target(joints_for_catching_to_object)
    robot.base.go()

    # Lower arm
    robot.tf_listener.waitForTransform("map", "hand_palm_link", rospy.Time(0),rospy.Duration(4.0))
    transform = robot.tf_listener.lookupTransform("map", "hand_palm_link", rospy.Time(0))
    z_diff = transform[0][2] - obj.xyz_max[2]

    joints_for_lower_arm = robot.arm.get_current_joint_values()
    if joints_for_lower_arm[0] - z_diff > 0.:
        joints_for_lower_arm[0] -= z_diff
    else:
        joints_for_lower_arm[0] = lowest_arm_height
    print("lowest_arm_height: {}".format(lowest_arm_height))
    print("robot.arm.get_current_joint_values(): {}".format(robot.arm.get_current_joint_values()))
    print("joints_for_lower_arm: {}".format(joints_for_lower_arm))
    robot.arm.set_joint_value_target(joints_for_lower_arm)
    robot.arm.go()

    # Pick
    robot.close_hand()

    # Move arm up
    robot.arm.set_joint_value_target(joints_for_hovering)
    robot.arm.go()

    # Move back to init pose
    joints_for_going_back_to_init_pose_trans = robot.base.get_current_joint_values()
    joints_for_going_back_to_init_pose_trans[0] = joints_for_going_back_to_init_pose[0]
    joints_for_going_back_to_init_pose_trans[1] = joints_for_going_back_to_init_pose[1]
    robot.base.set_joint_value_target(joints_for_going_back_to_init_pose_trans)
    robot.base.go()

    joints_for_going_back_to_init_pose_rot = robot.base.get_current_joint_values()
    joints_for_going_back_to_init_pose_rot[2] = joints_for_going_back_to_init_pose[2]
    robot.base.set_joint_value_target(joints_for_going_back_to_init_pose_rot)
    robot.base.go()

    # Keep it close to your heart
    robot.move_arm_init()

    if robot.is_hand_fully_closed():
        return False

    return True


# In[ ]:


def map_xy_to_frame_xy(coord, target_frame):
    robot.tf_listener.waitForTransform(target_frame, "/map", rospy.Time(0),rospy.Duration(4.0))
    point=PointStamped()
    point.header.frame_id = "map"
    point.header.stamp =rospy.Time(0)
    point.point.x=coord[0]
    point.point.y=coord[1]
    p=robot.tf_listener.transformPoint(target_frame, point)
    return p.point.x, p.point.y


# In[ ]:


def put_object_down_at_place(obj, goal_point, height):
    # Move head to prevent arm movement failures
    robot.move_head_tilt(0.)

    # Save joints for initial pose
    joints_for_going_back_to_init_pose = robot.base.get_current_joint_values()

    # Put arm forward
    joints_for_placing_arm_above = [height] + [math.radians(a) for a in [-90., 0., -90., 0., 0.]]
    robot.arm.set_joint_value_target(joints_for_placing_arm_above)
    robot.arm.go()


    # Compute angle for robot base to face arm parallel direction between base_link and object
    o_x, o_y = map_xy_to_frame_xy(goal_point, "base_link")
    yaw = math.pi/2. - math.atan2(o_x, o_y)

    joints_for_facing_object = robot.base.get_current_joint_values()
    joints_for_facing_object[2] += yaw

    robot.base.set_joint_value_target(joints_for_facing_object)
    robot.base.go()

    # Compute translation for robot base to actually face the object
    a_x, a_y = robot.get_diff_between("base_link", "arm_flex_link")

    robot.tf_listener.waitForTransform("/base_link", "/odom", rospy.Time(0),rospy.Duration(4.0))
    point=PointStamped()
    point.header.frame_id = "base_link"
    point.header.stamp =rospy.Time(0)
    point.point.y= -a_y
    p=robot.tf_listener.transformPoint("odom", point)

    joints_for_going_to_object = robot.base.get_current_joint_values()
    joints_for_going_to_object[0] = p.point.y
    joints_for_going_to_object[1] = p.point.x

    robot.base.set_joint_value_target(joints_for_going_to_object)
    robot.base.go()

    # Compute translation for robot base to get the object and get it
    oo_x, oo_y = map_xy_to_frame_xy(goal_point, "odom")
    ho_x, ho_y = robot.get_diff_between("odom", "hand_palm_link")

    joints_for_catching_to_object = robot.base.get_current_joint_values()
    joints_for_catching_to_object[0] += oo_y - ho_y
    joints_for_catching_to_object[1] += oo_x - ho_x

    robot.base.set_joint_value_target(joints_for_catching_to_object)
    robot.base.go()

    # # Lower arm if necessary
    # robot.tf_listener.waitForTransform("map", "hand_palm_link", rospy.Time(0),rospy.Duration(4.0))
    # transform = robot.tf_listener.lookupTransform("map", "hand_palm_link", rospy.Time(0))
    # z_diff = transform[0][2] - obj.xyz_max[2]

    # joints_for_lower_arm_picking_from_ground = robot.arm.get_current_joint_values()
    # z_diff = 0. if joints_for_lower_arm_picking_from_ground[0] - z_diff < 0. else z_diff
    # joints_for_lower_arm_picking_from_ground = robot.arm.get_current_joint_values()
    # joints_for_lower_arm_picking_from_ground[0] -= z_diff
    # robot.arm.set_joint_value_target(joints_for_lower_arm_picking_from_ground)
    # robot.arm.go()

    # Place
    robot.open_hand()
    robot.shake_wrist()

    # Move arm up
    robot.arm.set_joint_value_target(joints_for_placing_arm_above)
    robot.arm.go()

    # Close hand, the object should long have fallen
    robot.close_hand()

    # Move back to init pose
    joints_for_going_back_to_init_pose_trans = robot.base.get_current_joint_values()
    joints_for_going_back_to_init_pose_trans[0] = joints_for_going_back_to_init_pose[0]
    joints_for_going_back_to_init_pose_trans[1] = joints_for_going_back_to_init_pose[1]
    robot.base.set_joint_value_target(joints_for_going_back_to_init_pose_trans)
    robot.base.go()

    joints_for_going_back_to_init_pose_rot = robot.base.get_current_joint_values()
    joints_for_going_back_to_init_pose_rot[2] = joints_for_going_back_to_init_pose[2]
    robot.base.set_joint_value_target(joints_for_going_back_to_init_pose_rot)
    robot.base.go()

    # Set arm back to init
    robot.move_arm_init()


# In[ ]:


# "Tray_A", "Tray_B", "Container_A", "Container_B", "Drawer_top", "Drawer_bottom", "Drawer_left", "Bin_A", "Bin_B"
tray_a_counter = 0
tray_b_counter = 0

def choose_object_destination(obj):
    deposit_area_names = []
    if obj.label:
        deposit_area_names = message_parser.get_deposit(obj.label)

    if deposit_area_names:
        if "Bin_A" in deposit_area_names:
            return utils.IN_FRONT_BINS_GOAL, utils.BIN_A_1[0], utils.HEIGHT_ABOVE_BINS
        elif "Container_A" in deposit_area_names:
            return utils.IN_FRONT_DEPOSIT_TABLE_GOAL, utils.CONTAINER_A_1[0], utils.HEIGHT_ABOVE_CONTAINER_A
        elif "Tray_A" in deposit_area_names or "Tray_B" in deposit_area_names:
            if len(deposit_area_names) == 1:
                if deposit_area_names[0] == "Tray_A":
                    return utils.IN_FRONT_DEPOSIT_TABLE_GOAL, utils.TRAY_A_1[0], utils.HEIGHT_ABOVE_TRAYS
                elif deposit_area_names[0] == "Tray_B":
                    return utils.IN_FRONT_DEPOSIT_TABLE_GOAL, utils.TRAY_B_1[0], utils.HEIGHT_ABOVE_TRAYS
            else:
                if tray_a_counter < tray_b_counter:
                    return utils.IN_FRONT_DEPOSIT_TABLE_GOAL, utils.TRAY_A_1[0], utils.HEIGHT_ABOVE_TRAYS
                else:
                    return utils.IN_FRONT_DEPOSIT_TABLE_GOAL, utils.TRAY_B_1[0], utils.HEIGHT_ABOVE_TRAYS

    # By default, everything goes to the BIN_B (black)
    return utils.IN_FRONT_BINS_GOAL, utils.BIN_B_1[0], utils.HEIGHT_ABOVE_BINS



# In[6]:


# time_watchdog_thread = threading.Thread(target=utils.time_watchdog, kwargs={"max_minutes": 19, "max_seconds": 50})
# time_watchdog_thread.start()
start_time = rospy.Time.now()

previous_convex_footprints = []

area_params = [
    {
        "observation_goal": utils.IN_FRONT_LARGE_TABLE_GROUND_OBJECTS_GOAL,
        "observation_tilt": -0.85,
        "joints_for_hovering": [0.4] + [math.radians(a) for a in [-107., 0., -73., 0., 0.]],
        "lowest_arm_height": 0.,
        "pose_z_min": 0.,
        "pose_z_max": utils.LARGE_TABLE_HEIGHT,
        "xy_polygon": utils.GROUND_OBJECTS_AREA
    },
    {
        "observation_goal": utils.IN_FRONT_SMALL_TABLE_GROUND_OBJECTS_GOAL,
        "observation_tilt": -0.85,
        "joints_for_hovering": [0.4] + [math.radians(a) for a in [-107., 0., -73., 0., 0.]],
        "lowest_arm_height": 0.,
        "pose_z_min": 0.,
        "pose_z_max": utils.LARGE_TABLE_HEIGHT,
        "xy_polygon": utils.GROUND_OBJECTS_AREA
    },
    {
        "observation_goal": utils.CLOSER_TO_LARGE_TABLE_GOAL,
        "observation_tilt": -0.5,
        "joints_for_hovering": [0.59] + [math.radians(a) for a in [-90., 0., -90., 0., 0.]],
        "lowest_arm_height": 0.33,
        "pose_z_min": utils.LARGE_TABLE_HEIGHT,
        "pose_z_max": utils.LARGE_TABLE_HEIGHT + 1.,
        "xy_polygon": utils.LARGE_TABLE_OBJECTS_AREA
    },
    {
        "observation_goal": utils.CLOSER_TO_SMALL_TABLE_GOAL,
        "observation_tilt": -0.5,
        "joints_for_hovering": [0.69] + [math.radians(a) for a in [-90., 0., -90., 0., 0.]],
        "lowest_arm_height": 0.53,
        "pose_z_min": utils.SMALL_TABLE_HEIGHT,
        "pose_z_max": utils.SMALL_TABLE_HEIGHT + 1.,
        "xy_polygon": utils.SMALL_TABLE_OBJECTS_AREA
    }
]

params = area_params.pop(0)

while True:
    duration = (rospy.Time.now() - start_time).secs
    if duration > 13 * 60:
        break

    rospy.loginfo("Moving to observation point.")
    robot.move_base_actual_goal(params["observation_goal"])
    rospy.loginfo("Moved to observation point.")
    robot.move_head_tilt(params["observation_tilt"])
    rospy.loginfo("Observing...")
    current_objects = scene.wait_for_one_detection(use_labels=True)
    rospy.loginfo("Observed: {}".format(
        str([obj.name + " - " + (obj.label if obj.label else "NO LABEL") for obj in current_objects.values()])
    ))
    obj, uid_by_distance = get_chosen_object(current_objects, previous_convex_footprints, params["pose_z_min"], params["pose_z_max"], params["xy_polygon"])
    if obj:
        rospy.loginfo("Chosen object is: {} - {}".format(obj.name, (obj.label if obj.label else "NO LABEL")))
        if isinstance(obj.convex_footprint, Polygon):
            previous_convex_footprints.append(obj.convex_footprint)
        else:
            previous_convex_footprints.append(obj.convex_footprint.buffer(0.05))
        is_objet_picked = pick_object_away(obj, params["joints_for_hovering"], params["lowest_arm_height"])
        if not is_objet_picked:
            rospy.loginfo("Object {} - {}: FAILED PICKING.".format(obj.name, (obj.label if obj.label else "NO LABEL")))
            continue
        else:
            rospy.loginfo("Object {} - {}: SUCCESSFULLY PICKED.".format(obj.name, (obj.label if obj.label else "NO LABEL")))
            robot.move_head_tilt(0.)

            nav_goal, goal_point, arm_height = choose_object_destination(obj)
            rospy.loginfo("Object {} - {}: MOVING TO {}, GOAL POINT {}, ARM HEIGHT {}.".format(
                obj.name, (obj.label if obj.label else "NO LABEL"), str(nav_goal), str(goal_point), str(arm_height)
            ))

            robot.move_base_actual_goal(nav_goal)
            rospy.loginfo("Object {} - {}: MOVED TO {}.".format(obj.name, (obj.label if obj.label else "NO LABEL"), str(nav_goal)))

            put_object_down_at_place(obj, goal_point, arm_height)
            rospy.loginfo("Object {} - {}: PUT DOWN AT {}.".format(obj.name, (obj.label if obj.label else "NO LABEL"), str(goal_point)))

            if goal_point == utils.TRAY_A_1[0]:
                tray_a_counter += 1
            elif goal_point == utils.TRAY_B_1[0]:
                tray_b_counter += 1
    else:
        rospy.loginfo("No object to move could be found.")
        try:
            params = area_params.pop(0)
        except Exception:
            break


# In[ ]:


###########################


# In[ ]:


###########################


# In[ ]:


###########################


# In[ ]:


###########################


# In[ ]:


###########################


# In[6]:


robot.move_base_actual_goal(utils.BESIDES_BIN_GOAL)
robot.move_base_actual_goal(utils.BESIDES_BINS_TURN_GOAL)
robot.move_base_actual_goal(utils.OBSTACLE_AVOIDANCE_AREA_GOAL)


# In[7]:


robot.move_head_tilt(-1)


# In[8]:


current_objects = scene.wait_for_one_detection()


# In[9]:


def get_sorted_obj_list_by_distance(cur_objects):
    robot.tf_listener.waitForTransform("map", "base_link", rospy.Time(0),rospy.Duration(4.0))
    robot_transform = robot.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
    robot_pose_in_map = robot_transform[0][0], robot_transform[0][1], math.degrees(tf.transformations.euler_from_quaternion(robot_transform[1])[2])

    uid_by_distance = []
    uid_to_convex_footprint = {}
    for uid, obj in cur_objects.items():
        convex_footprint = MultiPoint(obj.bb_coords_2d).convex_hull
        if convex_footprint.intersects(utils.TABOO_AREA_POLYGON):
            min_distance = float("inf")
            for coord in obj.bb_coords_2d:
                min_distance = min(min_distance, utils.euclidean_distance(coord, robot_pose_in_map))
            uid_by_distance.append((uid, min_distance))
            uid_to_convex_footprint[uid] = convex_footprint
    uid_by_distance = sorted(uid_by_distance, key=lambda tup: tup[1])
    return uid_by_distance


# In[10]:


def pick_object_away(obj):
    # Compute angle for robot base to face arm parallel direction between base_link and object
    o_x, o_y = robot.get_diff_between("base_link", obj.name)
    yaw = math.pi/2. - math.atan2(o_x, o_y)

    joints_for_facing_object = robot.base.get_current_joint_values()
    joints_for_facing_object[2] += yaw

    robot.base.set_joint_value_target(joints_for_facing_object)
    robot.base.go()

    # Set to picking pose
    joints_for_arm_picking_from_ground = [0.1] + [math.radians(a) for a in [-107., 0., -73., 0., 0.]]
    robot.arm.set_joint_value_target(joints_for_arm_picking_from_ground)
    robot.arm.go()
    robot.open_hand()

    # Compute translation for robot base to actually face the object
    a_x, a_y = robot.get_diff_between("base_link", "arm_flex_link")

    robot.tf_listener.waitForTransform("/base_link", "/odom", rospy.Time(0),rospy.Duration(4.0))
    point=PointStamped()
    point.header.frame_id = "base_link"
    point.header.stamp =rospy.Time(0)
    point.point.y= -a_y
    p=robot.tf_listener.transformPoint("odom", point)

    joints_for_going_to_object = robot.base.get_current_joint_values()
    joints_for_going_to_object[0] = p.point.y
    joints_for_going_to_object[1] = p.point.x

    robot.base.set_joint_value_target(joints_for_going_to_object)
    robot.base.go()

    # Compute translation for robot base to get the object and get it
    oo_x, oo_y = robot.get_diff_between("odom", obj.name)
    ho_x, ho_y = robot.get_diff_between("odom", "hand_palm_link")

    joints_for_catching_to_object = robot.base.get_current_joint_values()
    joints_for_catching_to_object[0] += oo_y - ho_y
    joints_for_catching_to_object[1] += oo_x - ho_x

    robot.base.set_joint_value_target(joints_for_catching_to_object)
    robot.base.go()

    # Lower arm
    joints_for_lower_arm_picking_from_ground = robot.arm.get_current_joint_values()
    joints_for_lower_arm_picking_from_ground[0] = 0.
    robot.arm.set_joint_value_target(joints_for_lower_arm_picking_from_ground)
    robot.arm.go()

    # Pick
    robot.close_hand()

    # Move arm up
    robot.arm.set_joint_value_target(joints_for_arm_picking_from_ground)
    robot.arm.go()

    # Keep it close to your heart
    robot.move_arm_init()

    if robot.is_hand_fully_closed():
        return False

    # Turn 180deg
    joints_turn_180_deg = robot.base.get_current_joint_values()
    joints_turn_180_deg[2] -= math.radians(180)
    robot.base.set_joint_value_target(joints_turn_180_deg)
    robot.base.go()

    # Deliver
    robot.arm.set_joint_value_target(joints_for_arm_picking_from_ground)
    robot.arm.go()
    robot.open_hand()
    robot.shake_wrist()

    # Reset arm pose
    robot.move_arm_init()
    robot.close_hand()

    # Turn 180deg again
    joints_turn_180_deg = robot.base.get_current_joint_values()
    joints_turn_180_deg[2] += math.radians(180)
    robot.base.set_joint_value_target(joints_turn_180_deg)
    robot.base.go()

    return True


# In[11]:


uid_by_distance = get_sorted_obj_list_by_distance(current_objects)


# In[12]:


# Initial base joints
joints_for_going_back = robot.base.get_current_joint_values()

is_object_moved = True
for (uid, _) in uid_by_distance:
    obj = current_objects[uid]
    is_object_moved = pick_object_away(obj)
    joints_for_going_back = robot.base.get_current_joint_values()
    if not is_object_moved:
        break
if not is_object_moved:
    robot.base.set_joint_value_target(joints_for_going_back)
    robot.base.go()
    current_objects = scene.wait_for_one_detection()
    uid_by_distance = get_sorted_obj_list_by_distance(current_objects)
    for (uid, _) in uid_by_distance:
        obj = current_objects[uid]
        pick_object_away(obj)


# In[13]:


robot.move_arm_init()
robot.close_hand()


# In[14]:


robot.move_head_tilt(-0.9)


# In[15]:


enter_room_02_goal_str = '{"header": {"stamp": {"secs": 688, "nsecs": 512000000}, "frame_id": "", "seq": 11}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 688, "nsecs": 512000000}, "frame_id": "map", "seq": 11}, "pose": {"position": {"y": 2.9992051124572754, "x": 2.3737993240356445, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.7056854446361143, "w": 0.708525266471655}}}}}'
enter_room_02_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', enter_room_02_goal_str).goal


# In[16]:


robot.move_base_actual_goal(enter_room_02_goal)


# In[17]:


in_front_shelf_goal_str = '{"header": {"stamp": {"secs": 607, "nsecs": 362000000}, "frame_id": "", "seq": 6}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 607, "nsecs": 353000000}, "frame_id": "map", "seq": 6}, "pose": {"position": {"y": 3.7436118125915527, "x": 2.2750515937805176, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.7071067966408575, "w": 0.7071067657322372}}}}}'
in_front_shelf_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', in_front_shelf_goal_str).goal


# In[18]:


robot.move_base_actual_goal(in_front_shelf_goal)


# In[19]:


robot.move_head_tilt(-0.2)


# In[20]:


current_objects = scene.wait_for_one_detection(use_labels=True)


# In[21]:


def get_chosen_object(cur_objects):

    chosen_object = None

    # Prioritize choosing objects that look like the required one
    required_label = message_parser.get_object_darknet()
    if required_label:
        rospy.loginfo("Object to be delivered is: {}".format(required_label))

    # Choose closest object that fits in robot's hand by default otherwise
    uid_by_distance = []
    for uid, obj in cur_objects.items():
        convex_footprint = MultiPoint(obj.bb_coords_2d).convex_hull
        if isinstance(convex_footprint, Polygon):
            obj_radius = utils.get_circumscribed_radius(convex_footprint)
        else:
            obj_radius = 0.00000000001
        if obj_radius <= robot.GRASP_RADIUS:
            x, _= robot.get_diff_between("base_link", obj.name)
            uid_by_distance.append((uid, x))
            if required_label and obj.label == required_label:
                chosen_object = obj

    if not chosen_object:
        uid_by_distance = sorted(uid_by_distance, key=lambda tup: tup[1])
        if uid_by_distance:
            chosen_object = cur_objects[uid_by_distance[0][0]]

    if not chosen_object:
        rospy.logwarn("No object was able to be chosen. Stopping robot.")
        sys.exit(0)

    return chosen_object


# In[22]:


chosen_object = get_chosen_object(current_objects)


# In[23]:


FIRST_SHELF_LINEAR_JOINT_HEIGHT = 0.21
SECOND_SHELF_LINEAR_JOINT_HEIGHT = 0.51
SECOND_SHELF_HEIGHT = 0.78

def pick_object_from_shelf(obj):
    # Identify related shelf
    linear_joint_height = SECOND_SHELF_LINEAR_JOINT_HEIGHT if (obj.xyz_med[2] >= SECOND_SHELF_HEIGHT) else FIRST_SHELF_LINEAR_JOINT_HEIGHT

    # Save joints for pose in front of shelf
    joints_for_going_back_in_front_shelf = robot.base.get_current_joint_values()

    # Open hand and go to straight arm joints
    robot.open_hand()
    straight_arm = [linear_joint_height] + [math.radians(a) for a in [-90., 0., 0., 0., 0.]]
    robot.arm.set_joint_value_target(straight_arm)
    robot.arm.go()

    # Move parallel direction from base link to object
    diff_x, diff_y = robot.get_diff_between("base_link", obj.name)
    yaw = math.pi/2. - math.atan2(diff_x, diff_y)
    math.degrees(yaw)
    joints_for_facing_object = robot.base.get_current_joint_values()
    joints_for_facing_object[2] += yaw
    robot.base.set_joint_value_target(joints_for_facing_object)
    robot.base.go()

    # Translate in front of object
    a_x, a_y = robot.get_diff_between("base_link", "arm_flex_link")
    robot.tf_listener.waitForTransform("/base_link", "/odom", rospy.Time(0),rospy.Duration(4.0))
    point=PointStamped()
    point.header.frame_id = "base_link"
    point.header.stamp =rospy.Time(0)
    point.point.y=-a_y
    p=robot.tf_listener.transformPoint("odom", point)
    joints_for_going_to_object = robot.base.get_current_joint_values()
    joints_for_going_to_object[0] = p.point.y
    joints_for_going_to_object[1] = p.point.x
    robot.base.set_joint_value_target(joints_for_going_to_object)
    robot.base.go()

    # Translate to object
#     obj_o_x, obj_o_y = robot.get_diff_between("odom", obj.name)
#     print("obj_o_x, obj_o_y: {}, {}".format(obj_o_x, obj_o_y))
    r_x, r_y = robot.get_diff_between("map", "hand_palm_link")
#     print("r_x, r_y: {}, {}".format(r_x, r_y))
    min_distance_to_robot = float("inf")
    nearest_o_x, nearest_o_y = robot.get_diff_between("map", obj.name)
#     print("nearest_o_x, nearest_o_y: {}, {}".format(nearest_o_x, nearest_o_y))
    for pixel in obj.pixels:
        x, y, z = pixel.x, pixel.y, pixel.z
        dist = utils.euclidean_distance((r_x, r_y), (x, y))
        if dist < min_distance_to_robot:
            min_distance_to_robot = dist
            nearest_o_x, nearest_o_y, nearest_o_z = x, y, z
#     print("nearest_o_x, nearest_o_y, nearest_o_z: {}, {}".format(nearest_o_x, nearest_o_y, nearest_o_z))


    robot.tf_listener.waitForTransform("/odom", "/map", rospy.Time(0),rospy.Duration(4.0))
    point=PointStamped()
    point.header.frame_id = "map"
    point.header.stamp =rospy.Time(0)
    point.point.x=nearest_o_x
    point.point.y=nearest_o_y
    point.point.z=nearest_o_z
    p=robot.tf_listener.transformPoint("odom", point)
    obj_o_x, obj_o_y = p.point.x, p.point.y
#     print("obj_o_x, obj_o_y: {}, {}".format(obj_o_x, obj_o_y))
    ######

    ho_x, ho_y = robot.get_diff_between("odom", "hand_palm_link")
    joints_for_catching_to_object = robot.base.get_current_joint_values()
    joints_for_catching_to_object[0] += obj_o_y - ho_y
    joints_for_catching_to_object[1] += obj_o_x - ho_x
    robot.base.set_joint_value_target(joints_for_catching_to_object)
    robot.base.go()

    # Pick it
    robot.close_hand()

    # Lift it slightly
    joints_for_lifting_object = robot.arm.get_current_joint_values()
    joints_for_lifting_object[0] += 0.01
    robot.arm.set_joint_value_target(joints_for_lifting_object)
    robot.arm.go()

    # Move back in front of shelf
    joints_for_going_back_in_front_shelf_trans = robot.base.get_current_joint_values()
    joints_for_going_back_in_front_shelf_trans[0] = joints_for_going_back_in_front_shelf[0]
    joints_for_going_back_in_front_shelf_trans[1] = joints_for_going_back_in_front_shelf[1]
    robot.base.set_joint_value_target(joints_for_going_back_in_front_shelf_trans)
    robot.base.go()
    joints_for_going_back_in_front_shelf_rot = robot.base.get_current_joint_values()
    joints_for_going_back_in_front_shelf_rot[2] = joints_for_going_back_in_front_shelf[2]
    robot.base.set_joint_value_target(joints_for_going_back_in_front_shelf_rot)
    robot.base.go()

    # Keep object close to your heart
    robot.move_arm_init()

    if robot.is_hand_fully_closed():
        return False

    return True


# In[24]:


is_pick_success = pick_object_from_shelf(chosen_object)
if not is_pick_success:
    current_objects = scene.wait_for_one_detection(use_labels=True)
    chosen_object = get_chosen_object(current_objects)
    is_pick_success = pick_object_from_shelf(chosen_object)


# In[25]:


move_between_humans_goal_str = '{"header": {"stamp": {"secs": 134, "nsecs": 703000000}, "frame_id": "", "seq": 0}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 134, "nsecs": 679000000}, "frame_id": "map", "seq": 0}, "pose": {"position": {"y": 3.857577323913574, "x": 1.0511448383331299, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.9999999998344654, "w": -1.819530991026369e-05}}}}}'
move_between_humans_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', move_between_humans_goal_str).goal


# In[26]:


robot.move_base_actual_goal(move_between_humans_goal)


# In[27]:


latest_human_side_instruction = message_parser.get_person()
if latest_human_side_instruction:
    rospy.loginfo("Object must be delivered to human: {}".format(latest_human_side_instruction))

if latest_human_side_instruction == "right":
    in_front_human_right_goal_str = '{"header": {"stamp": {"secs": 176, "nsecs": 562000000}, "frame_id": "", "seq": 1}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 176, "nsecs": 562000000}, "frame_id": "map", "seq": 1}, "pose": {"position": {"y": 3.909142017364502, "x": 0.40349310636520386, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.9999394114821857, "w": -0.011007877391217903}}}}}'
    in_front_human_right_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', in_front_human_right_goal_str).goal
    robot.move_base_actual_goal(in_front_human_right_goal)

else:
    in_front_human_left_goal_str = '{"header": {"stamp": {"secs": 209, "nsecs": 613000000}, "frame_id": "", "seq": 2}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 209, "nsecs": 613000000}, "frame_id": "map", "seq": 2}, "pose": {"position": {"y": 2.8555641174316406, "x": 0.5514420866966248, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.9999989738094117, "w": -0.0014326130403864133}}}}}'
    in_front_human_left_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', in_front_human_left_goal_str).goal
    robot.move_base_actual_goal(in_front_human_left_goal)


# In[28]:


robot.move_arm_neutral()


# In[ ]:


# Fully exit code to stop simulation otherwise runs get too long
os._exit(1)


# In[ ]:
