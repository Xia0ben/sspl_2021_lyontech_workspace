#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic(u'script', u'bash', u'sudo apt-get update && sudo apt-get install -y ros-melodic-rospy-message-converter')


# In[ ]:


get_ipython().run_cell_magic(u'script', u'bash', u'pip install scipy scikit-learn colour shapely aabbtree future matplotlib opencv-contrib-python==4.0.0.21')


# In[ ]:


get_ipython().run_cell_magic(u'script', u'bash --bg', u'rviz -d /workspace/notebooks/data/3_navigation.rviz > /dev/null 2>&1')


# In[1]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[2]:


import math

import tf

import sys

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

from shapely.geometry import MultiPoint, Polygon


import utils

robot = utils.Robot()
scene = utils.Scene(start_on_init=False)
message_parser = utils.MessageParser()

rospy.loginfo("Imports done, robot initialized.")


# In[3]:


utils.NavGoalToJsonFileSaver("saved_msg.json")


# In[4]:


with open("saved_msg.json") as f:
    print(f.read())


# In[5]:


beside_bins_goal_str = '{"header": {"stamp": {"secs": 182, "nsecs": 889000000}, "frame_id": "", "seq": 1}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 182, "nsecs": 889000000}, "frame_id": "map", "seq": 1}, "pose": {"position": {"y": 0.31022635102272034, "x": 2.4421634674072266, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": -0.0026041090858226357, "w": 0.9999966093021861}}}}}' 
beside_bins_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', beside_bins_goal_str).goal

rospy.loginfo("Sending first goal")
robot.move_base_actual_goal(beside_bins_goal)
rospy.loginfo("First goal sent")

beside_bins_turn_goal_str = '{"header": {"stamp": {"secs": 208, "nsecs": 770000000}, "frame_id": "", "seq": 2}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 208, "nsecs": 743000000}, "frame_id": "map", "seq": 2}, "pose": {"position": {"y": 0.4013778567314148, "x": 2.4725470542907715, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.7055942189706708, "w": 0.7086161148006508}}}}}' 
beside_bins_turn_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', beside_bins_turn_goal_str).goal

robot.move_base_actual_goal(beside_bins_turn_goal)


# In[6]:


obstacle_avoidance_area_goal_str = '{"header": {"stamp": {"secs": 1218, "nsecs": 867000000}, "frame_id": "", "seq": 21}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 1218, "nsecs": 867000000}, "frame_id": "map", "seq": 21}, "pose": {"position": {"y": 1.7440035343170166, "x": 2.618055582046509, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.7167735161966976, "w": 0.697306049363565}}}}}'
obstacle_avoidance_area_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', obstacle_avoidance_area_goal_str).goal

robot.move_base_actual_goal(obstacle_avoidance_area_goal)


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


uid_by_distance = get_sorted_obj_list_by_distance(current_objects)


# In[11]:


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


# In[29]:


# To save points published in Rviz, simply use the following commands


# In[30]:


# saver = PointsSaver()


# In[ ]:


# coords = saver.get_coords()
# coords


# In[ ]:


# To transform saved points in the base_link frame, simply use the following commands


# In[ ]:


# robot.tf_listener.waitForTransform("map", "base_link", rospy.Time(0),rospy.Duration(4.0))
# transform = robot.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
# current_pose = transform[0][0], transform[0][1], math.degrees(tf.transformations.euler_from_quaternion(transform[1])[2])
# transformed_coords = []
# for coord in saved_robot_coords:
#     point=PointStamped()
#     point.header.frame_id = "map"
#     point.header.stamp =rospy.Time(0)
#     point.point.x=coord[0]
#     point.point.y=coord[1]
#     p=robot.tf_listener.transformPoint("base_link", point)
#     transformed_coords.append((p.point.x, p.point.y))
# transformed_coords


# In[ ]:


robot.open_hand()


# In[ ]:


robot.gripper.get_current_joint_values()


# In[ ]:


import numpy as np



# In[ ]:




