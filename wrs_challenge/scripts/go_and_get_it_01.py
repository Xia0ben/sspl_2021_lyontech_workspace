#!/usr/bin/env python
# coding: utf-8

# ### Dependencies to add :
# 
# #### apt
# 
# ros-melodic-rospy-message-converter
# 
# #### pip
# 
# scipy
# scikit-learn
# shapely
# aabbtree

# In[1]:


get_ipython().run_cell_magic(u'script', u'bash --bg', u'rviz -d /workspace/notebooks/data/3_navigation.rviz > /dev/null 2>&1')


# In[15]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[16]:


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


# In[19]:


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


import collision

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
    unit_rotation = collision.Rotation(unit_angle, (robot_pose_in_map[0], robot_pose_in_map[1]))

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
               
        robot_collides, _, _, _, _, _ = collision.csv_check_collisions(
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


# In[ ]:


robot.move_head_tilt(-0.2)


# In[ ]:


joints_for_reaching_apple = [0.] + [math.radians(a) for a in [-50., 0., -40., 0., 0.]]
robot.arm.set_joint_value_target(joints_for_reaching_apple)
robot.arm.go()


# In[ ]:


robot.open_hand()


# In[ ]:


apple_x, apple_y = robot.get_diff_between("base_link", "object_with_hue_252.0")
yaw = math.pi/2. - math.atan2(apple_x, apple_y)
math.degrees(yaw)


# In[ ]:


joints_for_facing_apple = robot.base.get_current_joint_values()
joints_for_facing_apple[2] += yaw
joints_for_facing_apple


# In[ ]:


robot.base.set_joint_value_target(joints_for_facing_apple)
robot.base.go()


# In[ ]:


a_x, a_y = robot.get_diff_between("base_link", "arm_flex_link")
a_x, a_y


# In[ ]:


robot.tf_listener.waitForTransform("/base_link", "/odom", rospy.Time(0),rospy.Duration(4.0))
point=PointStamped()
point.header.frame_id = "base_link"
point.header.stamp =rospy.Time(0)
point.point.y=-a_y - 0.02  #Hardcoded compensation for overshoots
p=robot.tf_listener.transformPoint("odom", point)
p


# In[ ]:


joints_for_going_to_object = robot.base.get_current_joint_values()
joints_for_going_to_object[0] = p.point.y
joints_for_going_to_object[1] = p.point.x
joints_for_going_to_object


# In[ ]:


robot.base.set_joint_value_target(joints_for_going_to_object)
robot.base.go()


# In[ ]:


apple_o_x, apple_o_y = robot.get_diff_between("odom", "object_with_hue_252.0")
ho_x, ho_y = robot.get_diff_between("odom", "hand_palm_link")


# In[ ]:


joints_for_catching_to_object = robot.base.get_current_joint_values()
joints_for_catching_to_object[0] += apple_o_y - ho_y
joints_for_catching_to_object[1] += apple_o_x - ho_x
joints_for_catching_to_object


# In[ ]:


robot.base.set_joint_value_target(joints_for_catching_to_object)
robot.base.go()


# In[ ]:


robot.close_hand()


# In[ ]:


robot.base.set_joint_value_target(joints_for_going_to_object)
robot.base.go()


# In[ ]:


joints_for_taking_object_out = [
    0.,
    -0.37538909090912753,
    0.02490062095512746,
    -1.3316079179879008,
    -0.0109236366592258,
    0.
]


# In[ ]:


robot.arm.set_joint_value_target(joints_for_taking_object_out)
robot.arm.go()


# In[ ]:


robot.move_arm_neutral()


# In[ ]:


robot.move_arm_init()


# In[ ]:


turn_away_from_shelf_goal_str = '{"header": {"stamp": {"secs": 1194, "nsecs": 756000000}, "frame_id": "", "seq": 0}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 1194, "nsecs": 756000000}, "frame_id": "map", "seq": 0}, "pose": {"position": {"y": 4.0555524826049805, "x": 2.27242374420166, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.999989479138774, "w": -0.004587113663660934}}}}}'
turn_away_from_shelf_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', turn_away_from_shelf_goal_str).goal


# In[ ]:


robot.move_base_actual_goal(turn_away_from_shelf_goal)


# In[ ]:


move_between_humans_goal_str = '{"header": {"stamp": {"secs": 1342, "nsecs": 239000000}, "frame_id": "", "seq": 5}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 1342, "nsecs": 230000000}, "frame_id": "map", "seq": 5}, "pose": {"position": {"y": 3.354616641998291, "x": 0.4766915440559387, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.999855500851024, "w": 0.016999335809019533}}}}}'
move_between_humans_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', move_between_humans_goal_str).goal


# In[ ]:


robot.move_base_actual_goal(move_between_humans_goal)


# In[ ]:


instruction_listener = utils.InstructionListener()


# In[ ]:


rospy.sleep(1.)

latest_human_side_instruction = instruction_listener.get_latest_human_side_instruction()
if latest_human_side_instruction == "right":
    turn_to_human_right_goal_str = '{"header": {"stamp": {"secs": 1725, "nsecs": 663000000}, "frame_id": "", "seq": 7}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 1725, "nsecs": 657000000}, "frame_id": "map", "seq": 7}, "pose": {"position": {"y": 3.307887554168701, "x": 0.4466514587402344, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.8767583591536536, "w": -0.48093115895540894}}}}}'
    turn_to_human_right_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', turn_to_human_right_goal_str).goal
    robot.move_base_actual_goal(turn_to_human_right_goal)
    
    in_front_human_right_goal_str = ''
    in_front_human_right_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', in_front_human_right_goal_str).goal
    robot.move_base_actual_goal(in_front_human_right_goal)
    
elif latest_human_side_instruction == "left":
    turn_to_human_left_goal_str = '{"header": {"stamp": {"secs": 1713, "nsecs": 756000000}, "frame_id": "", "seq": 6}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 1713, "nsecs": 756000000}, "frame_id": "map", "seq": 6}, "pose": {"position": {"y": 3.334589719772339, "x": 0.4433136284351349, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.8656726416348606, "w": 0.5006105048088003}}}}}'
    turn_to_human_left_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', turn_to_human_left_goal_str).goal
    robot.move_base_actual_goal(turn_to_human_left_goal)
    
    in_front_human_left_goal_str = '{"header": {"stamp": {"secs": 2013, "nsecs": 108000000}, "frame_id": "", "seq": 11}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 2013, "nsecs": 105000000}, "frame_id": "map", "seq": 11}, "pose": {"position": {"y": 2.8104236125946045, "x": 0.7230702042579651, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.9999938547515791, "w": -0.0035057751037049652}}}}}'
    in_front_human_left_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', in_front_human_left_goal_str).goal
    robot.move_base_actual_goal(in_front_human_left_goal)
else:
    pass


# In[ ]:


robot.move_arm_neutral()


# In[ ]:


joints_for_delivering_object = [
    0.11320954301629421,
    -2.620044420577912,
    3.124453377905327,
    -0.8492111265425875,
    3.147435050072273,
    0.0
]


# In[ ]:


robot.arm.set_joint_value_target(joints_for_delivering_object)
robot.arm.go()


# In[ ]:


robot.open_hand()


# In[ ]:


robot.move_arm_init()
robot.close_hand()


# In[ ]:


# To save points published in Rviz, simply use the following commands


# In[ ]:


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




