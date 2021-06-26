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

from shapely.geometry import MultiPoint, Polygon, Point


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


# in_front_drawers_goal_str = '{"header": {"stamp": {"secs": 69, "nsecs": 237000000}, "frame_id": "", "seq": 2}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 69, "nsecs": 228000000}, "frame_id": "map", "seq": 2}, "pose": {"position": {"y": 0.5999923944473267, "x": 0.24766463041305542, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": -0.7189345607475961, "w": 0.6950777635363263}}}}}'
# in_front_drawers_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', in_front_drawers_goal_str).goal
# robot.move_base_actual_goal(in_front_drawers_goal)


# In[6]:


# robot.open_hand()
# READY_FOR_BOTTOM_DRAWERS_ARM_JOINTS = [.25] + [math.radians(a) for a in [-150., 0., 60., 0., 0.]]
# robot.arm.set_joint_value_target(READY_FOR_BOTTOM_DRAWERS_ARM_JOINTS)
# robot.arm.go()


# In[7]:


# GRASP_RIGHT_BOTTOM_RIGHT_DRAWER_BASE_JOINTS = [0.14590200293468028, 0.09235241684400744, -1.4923558765834182]
# robot.base.set_joint_value_target(GRASP_RIGHT_BOTTOM_RIGHT_DRAWER_BASE_JOINTS)
# robot.base.go()
# robot.close_hand()


# In[8]:


# PULL_RIGHT_BOTTOM_RIGHT_DRAWER_BASE_JOINTS = [0.448629245701104, 0.08066860734930102, -1.511430367700255]
# robot.base.set_joint_value_target(PULL_RIGHT_BOTTOM_RIGHT_DRAWER_BASE_JOINTS)
# robot.base.go()
# robot.open_hand()


# In[9]:


def get_chosen_object(cur_objects, previous_convex_footprints):
    chosen_object = None
    # Choose closest object that fits in robot's hand by default otherwise
    uid_by_distance = []
    for uid, obj in cur_objects.items():
        if obj.circumscribed_radius <= robot.GRASP_RADIUS:
            intersects = False
            convex_footprint = obj.convex_footprint
            point = Point([obj.pose[0], obj.pose[1]])
            if point.intersects(utils.GROUND_OBJECTS_AREA):
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


# In[10]:


def pick_object_away(obj):
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
    joints_for_arm_picking_from_ground = [0.4] + [math.radians(a) for a in [-107., 0., -73., 0., 0.]]
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
    robot.tf_listener.waitForTransform("map", "hand_palm_link", rospy.Time(0),rospy.Duration(4.0))
    transform = robot.tf_listener.lookupTransform("map", "hand_palm_link", rospy.Time(0))
    z_diff = transform[0][2] - obj.xyz_max[2]

    joints_for_lower_arm_picking_from_ground = robot.arm.get_current_joint_values()
    z_diff = 0. if joints_for_lower_arm_picking_from_ground[0] - z_diff < 0. else z_diff
    joints_for_lower_arm_picking_from_ground = robot.arm.get_current_joint_values()
    joints_for_lower_arm_picking_from_ground[0] -= z_diff
    robot.arm.set_joint_value_target(joints_for_lower_arm_picking_from_ground)
    robot.arm.go()
    
    # Pick
    robot.close_hand()
    
    # Move arm up
    robot.arm.set_joint_value_target(joints_for_arm_picking_from_ground)
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


# In[11]:


def map_xy_to_frame_xy(coord, target_frame):
    robot.tf_listener.waitForTransform(target_frame, "/map", rospy.Time(0),rospy.Duration(4.0))
    point=PointStamped()
    point.header.frame_id = "map"
    point.header.stamp =rospy.Time(0)
    point.point.x=coord[0]
    point.point.y=coord[1]
    p=robot.tf_listener.transformPoint(target_frame, point)
    return p.point.x, p.point.y


# In[12]:


def put_object_down_at_place(obj, goal_point, height):
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
    
    # Move arm up
    robot.arm.set_joint_value_target(joints_for_placing_arm_above)
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

    # Set arm back to init
    robot.move_arm_init()


# In[13]:


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
    


# In[ ]:


utils.TimeWatchDogThread().start()

previous_convex_footprints = []
while True: 
    rospy.loginfo("Moving to observation point.")
    robot.move_base_actual_goal(utils.IN_FRONT_LARGE_TABLE_GROUND_OBJECTS_GOAL)
    rospy.loginfo("Moved to observation point.")
    robot.move_head_tilt(-0.85)
    rospy.loginfo("Observing...")
    current_objects = scene.wait_for_one_detection(use_labels=True)
    rospy.loginfo("Observed: {}".format(
        str([obj.name + " - " + (obj.label if obj.label else "NO LABEL") for obj in current_objects.values()])
    ))
    obj, uid_by_distance = get_chosen_object(current_objects, previous_convex_footprints)
    if obj:
        rospy.loginfo("Chosen object is: {} - {}".format(obj.name, (obj.label if obj.label else "NO LABEL")))
        if isinstance(obj.convex_footprint, Polygon):
            previous_convex_footprints.append(obj.convex_footprint)
        else:
            previous_convex_footprints.append(obj.convex_footprint.buffer(0.05))
        is_objet_picked = pick_object_away(obj)
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
        break


# In[ ]:




