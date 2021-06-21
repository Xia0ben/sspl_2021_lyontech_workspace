import tf


import rospy
rospy.init_node("go_and_get_it_01")


from rospy_message_converter import json_message_converter


from geometry_msgs.msg import Pose, PointStamped


from utils import *


obstacle_avoidance_area_goal_str = '{"header": {"stamp": {"secs": 1218, "nsecs": 867000000}, "frame_id": "", "seq": 21}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 1218, "nsecs": 867000000}, "frame_id": "map", "seq": 21}, "pose": {"position": {"y": 1.7440035343170166, "x": 2.618055582046509, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.7167735161966976, "w": 0.697306049363565}}}}}'
obstacle_avoidance_area_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', obstacle_avoidance_area_goal_str).goal


# In[6]:


move_base_actual_goal(obstacle_avoidance_area_goal)


# In[7]:


move_head_tilt(-1)


# In[8]:


detector = ColorBasedObjectDetector()


# In[9]:


joints_for_swiping = [0.] + [math.radians(a) for a in [-146., 0., 53., 0., 0.]]
arm.set_joint_value_target(joints_for_swiping)
arm.go()


# In[10]:


move_hand(1)


# In[11]:


o_x, o_y = get_diff_between("base_link", "object_with_hue_185.0")
yaw = math.pi/2. - math.atan2(o_x, o_y)
math.degrees(yaw)


# In[12]:


joints_for_facing_object = base.get_current_joint_values()
joints_for_facing_object[2] += yaw
joints_for_facing_object


# In[13]:


base.set_joint_value_target(joints_for_facing_object)
base.go()


# In[14]:


a_x, a_y = get_diff_between("base_link", "arm_flex_link")
a_x, a_y


# In[15]:


listener = tf.TransformListener()
listener.waitForTransform("/base_link", "/odom", rospy.Time(0),rospy.Duration(4.0))
point=PointStamped()
point.header.frame_id = "base_link"
point.header.stamp =rospy.Time(0)
point.point.y=-a_y - 0.02  #Hardcoded compensation for overshoots
p=listener.transformPoint("odom", point)
p


# In[16]:


joints_for_going_to_object = base.get_current_joint_values()
joints_for_going_to_object[0] = p.point.y
joints_for_going_to_object[1] = p.point.x
joints_for_going_to_object


# In[17]:


base.set_joint_value_target(joints_for_going_to_object)
base.go()


# In[18]:


oo_x, oo_y = get_diff_between("odom", "object_with_hue_185.0")
ho_x, ho_y = get_diff_between("odom", "hand_palm_link")


# In[19]:


joints_for_catching_to_object = base.get_current_joint_values()
joints_for_catching_to_object[0] += oo_y - ho_y
joints_for_catching_to_object[1] += oo_x - ho_x
joints_for_catching_to_object


# In[20]:


base.set_joint_value_target(joints_for_catching_to_object)
base.go()


# In[21]:


move_hand(0)


# In[22]:


joints_for_clearing_object = base.get_current_joint_values()
joints_for_clearing_object[2] += math.radians(-40.)
joints_for_clearing_object


# In[23]:


base.set_joint_value_target(joints_for_clearing_object)
base.go()


# In[24]:


move_hand(1)


# In[25]:


move_arm_init()


# In[26]:


move_hand(0)
move_head_tilt(-0.9)


# In[27]:


rotate_before_stear_clear_goal_str = '{"header": {"stamp": {"secs": 117, "nsecs": 345000000}, "frame_id": "", "seq": 0}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 117, "nsecs": 345000000}, "frame_id": "map", "seq": 0}, "pose": {"position": {"y": 1.9495398998260498, "x": 2.5787177085876465, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.8523102169808354, "w": 0.5230366086901387}}}}}'
rotate_before_stear_clear_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', rotate_before_stear_clear_goal_str).goal


# In[28]:


move_base_actual_goal(rotate_before_stear_clear_goal)


# In[29]:


enter_room_02_goal_str = '{"header": {"stamp": {"secs": 187, "nsecs": 455000000}, "frame_id": "", "seq": 1}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 187, "nsecs": 452000000}, "frame_id": "map", "seq": 1}, "pose": {"position": {"y": 3.10935378074646, "x": 1.6804301738739014, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.8766056986594742, "w": 0.4812093609622895}}}}}'
enter_room_02_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', enter_room_02_goal_str).goal


# In[30]:


move_base_actual_goal(enter_room_02_goal)


# In[31]:


in_front_shelf_goal_str = '{"header": {"stamp": {"secs": 265, "nsecs": 137000000}, "frame_id": "", "seq": 3}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 265, "nsecs": 128000000}, "frame_id": "map", "seq": 3}, "pose": {"position": {"y": 3.871192455291748, "x": 2.2148544788360596, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.7140042250431763, "w": 0.7001413904494529}}}}}'
in_front_shelf_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', in_front_shelf_goal_str).goal


# In[32]:


move_base_actual_goal(in_front_shelf_goal)


# In[33]:


move_head_tilt(-0.2)


# In[34]:


joints_for_reaching_apple = [0.] + [math.radians(a) for a in [-50., 0., -40., 0., 0.]]
arm.set_joint_value_target(joints_for_reaching_apple)
arm.go()


# In[35]:


move_hand(1)


# In[36]:


apple_x, apple_y = get_diff_between("base_link", "object_with_hue_252.0")
yaw = math.pi/2. - math.atan2(apple_x, apple_y)
math.degrees(yaw)


# In[37]:


joints_for_facing_apple = base.get_current_joint_values()
joints_for_facing_apple[2] += yaw
joints_for_facing_apple


# In[38]:


base.set_joint_value_target(joints_for_facing_apple)
base.go()


# In[39]:


a_x, a_y = get_diff_between("base_link", "arm_flex_link")
a_x, a_y


# In[40]:


listener = tf.TransformListener()
listener.waitForTransform("/base_link", "/odom", rospy.Time(0),rospy.Duration(4.0))
point=PointStamped()
point.header.frame_id = "base_link"
point.header.stamp =rospy.Time(0)
point.point.y=-a_y - 0.02  #Hardcoded compensation for overshoots
p=listener.transformPoint("odom", point)
p


# In[41]:


joints_for_going_to_object = base.get_current_joint_values()
joints_for_going_to_object[0] = p.point.y
joints_for_going_to_object[1] = p.point.x
joints_for_going_to_object


# In[42]:


base.set_joint_value_target(joints_for_going_to_object)
base.go()


# In[43]:


apple_o_x, apple_o_y = get_diff_between("odom", "object_with_hue_252.0")
ho_x, ho_y = get_diff_between("odom", "hand_palm_link")


# In[44]:


joints_for_catching_to_object = base.get_current_joint_values()
joints_for_catching_to_object[0] += apple_o_y - ho_y
joints_for_catching_to_object[1] += apple_o_x - ho_x
joints_for_catching_to_object


# In[45]:


base.set_joint_value_target(joints_for_catching_to_object)
base.go()


# In[46]:


move_hand(0)


# In[47]:


base.set_joint_value_target(joints_for_going_to_object)
base.go()


# In[48]:


joints_for_taking_object_out = [
    0.,
    -0.37538909090912753,
    0.02490062095512746,
    -1.3316079179879008,
    -0.0109236366592258,
    0.
]


# In[49]:


arm.set_joint_value_target(joints_for_taking_object_out)
arm.go()


# In[50]:


move_arm_neutral()


# In[51]:


move_arm_init()


# In[52]:


turn_away_from_shelf_goal_str = '{"header": {"stamp": {"secs": 1194, "nsecs": 756000000}, "frame_id": "", "seq": 0}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 1194, "nsecs": 756000000}, "frame_id": "map", "seq": 0}, "pose": {"position": {"y": 4.0555524826049805, "x": 2.27242374420166, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.999989479138774, "w": -0.004587113663660934}}}}}'
turn_away_from_shelf_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', turn_away_from_shelf_goal_str).goal


# In[53]:


move_base_actual_goal(turn_away_from_shelf_goal)


# In[54]:


move_between_humans_goal_str = '{"header": {"stamp": {"secs": 1342, "nsecs": 239000000}, "frame_id": "", "seq": 5}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 1342, "nsecs": 230000000}, "frame_id": "map", "seq": 5}, "pose": {"position": {"y": 3.354616641998291, "x": 0.4766915440559387, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.999855500851024, "w": 0.016999335809019533}}}}}'
move_between_humans_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', move_between_humans_goal_str).goal


# In[55]:


move_base_actual_goal(move_between_humans_goal)


# In[2]:


instruction_listener = InstructionListener()


# In[3]:


latest_human_side_instruction = instruction_listener.get_latest_human_side_instruction()
if latest_human_side_instruction == "right":
    turn_to_human_right_goal_str = '{"header": {"stamp": {"secs": 1725, "nsecs": 663000000}, "frame_id": "", "seq": 7}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 1725, "nsecs": 657000000}, "frame_id": "map", "seq": 7}, "pose": {"position": {"y": 3.307887554168701, "x": 0.4466514587402344, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.8767583591536536, "w": -0.48093115895540894}}}}}'
    turn_to_human_right_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', turn_to_human_right_goal_str).goal
    move_base_actual_goal(turn_to_human_right_goal)

    in_front_human_right_goal_str = ''
    in_front_human_right_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', in_front_human_right_goal_str).goal
    move_base_actual_goal(in_front_human_right_goal)

elif latest_human_side_instruction == "left":
    turn_to_human_left_goal_str = '{"header": {"stamp": {"secs": 1713, "nsecs": 756000000}, "frame_id": "", "seq": 6}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 1713, "nsecs": 756000000}, "frame_id": "map", "seq": 6}, "pose": {"position": {"y": 3.334589719772339, "x": 0.4433136284351349, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.8656726416348606, "w": 0.5006105048088003}}}}}'
    turn_to_human_left_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', turn_to_human_left_goal_str).goal
    move_base_actual_goal(turn_to_human_left_goal)

    in_front_human_left_goal_str = '{"header": {"stamp": {"secs": 2013, "nsecs": 108000000}, "frame_id": "", "seq": 11}, "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}, "goal": {"target_pose": {"header": {"stamp": {"secs": 2013, "nsecs": 105000000}, "frame_id": "map", "seq": 11}, "pose": {"position": {"y": 2.8104236125946045, "x": 0.7230702042579651, "z": 0.0}, "orientation": {"y": 0.0, "x": 0.0, "z": 0.9999938547515791, "w": -0.0035057751037049652}}}}}'
    in_front_human_left_goal = json_message_converter.convert_json_to_ros_message('move_base_msgs/MoveBaseActionGoal', in_front_human_left_goal_str).goal
    move_base_actual_goal(in_front_human_left_goal)
else:
    pass


# In[4]:


move_arm_neutral()


# In[5]:


joints_for_delivering_object = [
    0.11320954301629421,
    -2.620044420577912,
    3.124453377905327,
    -0.8492111265425875,
    3.147435050072273,
    0.0
]


# In[6]:


arm.set_joint_value_target(joints_for_delivering_object)
arm.go()


# In[7]:


move_hand(1)


# In[9]:


move_arm_init()
move_hand(0)


# In[ ]:
