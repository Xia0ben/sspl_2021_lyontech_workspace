#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt

import math

import tf


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

from shapely.geometry import MultiPoint


import utils

robot = utils.Robot()

rospy.loginfo("Imports done, robot initialized.")


# In[17]:


utils.NavGoalToJsonFileSaver("saved_msg.json")


# In[18]:


with open("saved_msg.json") as f:
    print(f.read())



beside_bins_goal_str = '{"header": {"stamp": {"secs": 282, "nsecs": 639000000}, "frame_id": "", "seq": 0}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 282, "nsecs": 633000000}, "frame_id": "map", "seq": 0}, "pose": {"position": {"y": 0.04305005073547363, "x": 2.4967446327209473, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.003784852844821431, "w": 0.9999928374188203}}}}}'
beside_bins_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', beside_bins_goal_str).goal

rospy.loginfo("Sending first goal")
robot.move_base_actual_goal(beside_bins_goal)
rospy.loginfo("First goal sent")

beside_bins_turn_goal_str = '{"header": {"stamp": {"secs": 320, "nsecs": 169000000}, "frame_id": "", "seq": 1}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 320, "nsecs": 166000000}, "frame_id": "map", "seq": 1}, "pose": {"position": {"y": 0.14174890518188477, "x": 2.5354907512664795, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.7007005461194897, "w": 0.7134554959265847}}}}}'
beside_bins_turn_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', beside_bins_turn_goal_str).goal

robot.move_base_actual_goal(beside_bins_turn_goal)

obstacle_avoidance_area_goal_str = '{"header": {"stamp": {"secs": 1218, "nsecs": 867000000}, "frame_id": "", "seq": 21}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 1218, "nsecs": 867000000}, "frame_id": "map", "seq": 21}, "pose": {"position": {"y": 1.7440035343170166, "x": 2.618055582046509, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.7167735161966976, "w": 0.697306049363565}}}}}'
obstacle_avoidance_area_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', obstacle_avoidance_area_goal_str).goal

robot.move_base_actual_goal(obstacle_avoidance_area_goal)


# In[20]:


robot.move_head_tilt(-1)


# In[21]:


scene = utils.Scene(start_on_init=False)
current_objects = scene.wait_for_one_detection()


# In[22]:


robot.tf_listener.waitForTransform("map", "base_link", rospy.Time(0),rospy.Duration(4.0))
robot_transform = robot.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
robot_pose_in_map = robot_transform[0][0], robot_transform[0][1], math.degrees(tf.transformations.euler_from_quaternion(robot_transform[1])[2])

objects_to_move = {}
uid_by_distance = []
uid_to_convex_footprint = {}
for uid, obj in current_objects.items():
    convex_footprint = MultiPoint(obj.bb_coords_2d).convex_hull
    if convex_footprint.intersects(utils.TABOO_AREA_POLYGON):
        objects_to_move[uid] = obj
        min_distance = float("inf")
        for coord in obj.bb_coords_2d:
            min_distance = min(min_distance, utils.euclidean_distance(coord, robot_pose_in_map))
        uid_by_distance.append((uid, min_distance))
        uid_to_convex_footprint[uid] = convex_footprint
uid_by_distance = sorted(uid_by_distance, key=lambda tup: tup[1])


# In[23]:

def swipe_object_away(obj, object_polygon_at_start, robot_polygon_at_start, robot_pose_in_map, unit_angle=-5., debug_display=True):
    # Compute angle for robot base to face arm parallel direction between base_link and object
    o_x, o_y = robot.get_diff_between("base_link", obj.name)
    yaw = math.pi/2. - math.atan2(o_x, o_y)

    joints_for_facing_object = robot.base.get_current_joint_values()
    joints_for_facing_object[2] += yaw

    robot.base.set_joint_value_target(joints_for_facing_object)
    robot.base.go()

    # Set to swiping pose
    robot.move_arm_to_swiping_pose()
    robot.open_hand()

    # Compute translation for robot base to actually face the object
    a_x, a_y = robot.get_diff_between("base_link", "arm_flex_link")

    robot.tf_listener.waitForTransform("/base_link", "/odom", rospy.Time(0),rospy.Duration(4.0))
    point=PointStamped()
    point.header.frame_id = "base_link"
    point.header.stamp =rospy.Time(0)
    point.point.y= -a_y - 0.02  # Hardcoded correction to avoid overshoot
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

    robot.close_hand()

    # Clear the object, and reset robot pose
    unit_rotation = utils.Rotation(unit_angle, (robot_pose_in_map[0], robot_pose_in_map[1]))

    total_angle = unit_angle
    prev_robot_polygon_after_rotation = robot_polygon_at_start
    robot_collides = False

    if debug_display:
        fig, ax = plt.subplots()
        ax.plot(*prev_robot_polygon_after_rotation.exterior.xy)
        ax.plot(*utils.RIGHT_WALL_POLYGON.exterior.xy)
        ax.plot(*utils.LEFT_WALL_POLYGON.exterior.xy)
        ax.plot(*utils.TABOO_AREA_POLYGON.exterior.xy)

    while total_angle > -180. and not robot_collides:  # and not object_collides:
        robot_polygon_after_rotation = unit_rotation.apply(prev_robot_polygon_after_rotation)

        if debug_display:
            ax.plot(*robot_polygon_after_rotation.exterior.xy)

        robot_collides, _, _, _, _, _ = utils.csv_check_collisions(
            0, utils.OTHER_POLYGONS, [prev_robot_polygon_after_rotation, robot_polygon_after_rotation], [unit_rotation],
            bb_type='minimum_rotated_rectangle', break_at_first=True
        )

        prev_robot_polygon_after_rotation = robot_polygon_after_rotation
        total_angle += unit_angle

    if debug_display:
        ax.axis('equal')
        fig.show()

    total_angle -= unit_angle if not(robot_collides or object_collides) else 2. * unit_angle
    joints_for_clearing_object = robot.base.get_current_joint_values()
    joints_for_clearing_object[2] += math.radians(total_angle)

    robot.base.set_joint_value_target(joints_for_clearing_object)
    robot.base.go()

    robot.open_hand()

    joints_for_raising_arm = robot.arm.get_current_joint_values()
    joints_for_raising_arm[0] += 0.4
    robot.arm.set_joint_value_target(joints_for_reaching_apple)
    robot.arm.go()




# In[24]:


robot_polygon = utils.set_polygon_pose(utils.ROBOT_POLYGON_FOR_SWIPING, (0., 0., 0.), robot_pose_in_map, (0., 0.))
for (uid, _) in uid_by_distance:
    obj = objects_to_move[uid]
    swipe_object_away(obj, uid_to_convex_footprint[uid], robot_polygon, robot_pose_in_map)


# In[25]:


robot.move_arm_init()
robot.close_hand()


# In[26]:


robot.move_head_tilt(-0.9)


# In[27]:


rotate_before_stear_clear_goal_str = '{"header": {"stamp": {"secs": 117, "nsecs": 345000000}, "frame_id": "", "seq": 0}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 117, "nsecs": 345000000}, "frame_id": "map", "seq": 0}, "pose": {"position": {"y": 1.9495398998260498, "x": 2.5787177085876465, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.8523102169808354, "w": 0.5230366086901387}}}}}'
rotate_before_stear_clear_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', rotate_before_stear_clear_goal_str).goal


# In[28]:


robot.move_base_actual_goal(rotate_before_stear_clear_goal)


# In[29]:


enter_room_02_goal_str = '{"header": {"stamp": {"secs": 187, "nsecs": 455000000}, "frame_id": "", "seq": 1}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 187, "nsecs": 452000000}, "frame_id": "map", "seq": 1}, "pose": {"position": {"y": 3.10935378074646, "x": 1.6804301738739014, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.8766056986594742, "w": 0.4812093609622895}}}}}'
enter_room_02_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', enter_room_02_goal_str).goal


# In[30]:


robot.move_base_actual_goal(enter_room_02_goal)


# In[ ]:


in_front_shelf_goal_str = '{"header": {"stamp": {"secs": 265, "nsecs": 137000000}, "frame_id": "", "seq": 3}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 265, "nsecs": 128000000}, "frame_id": "map", "seq": 3}, "pose": {"position": {"y": 3.871192455291748, "x": 2.2148544788360596, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.7140042250431763, "w": 0.7001413904494529}}}}}'
in_front_shelf_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', in_front_shelf_goal_str).goal


# In[ ]:


robot.move_base_actual_goal(in_front_shelf_goal)
