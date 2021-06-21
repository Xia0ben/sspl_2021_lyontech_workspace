#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tf

import rospy
rospy.init_node("go_and_get_it_01")

from rospy_message_converter import json_message_converter

from geometry_msgs.msg import Pose, PointStamped

import sys

from utils import *

if __name__=='__main__':

    try:
        # 視線を少し下げる
        move_head_tilt(-0.4)
    except:
        rospy.logerr('fail to init')
        sys.exit()

    try:
        # 移動姿勢
        move_arm_init()
        # 長テーブルの前に移動
        move_base_goal(1, 0.5, 90)
    except:
        rospy.logerr('fail to move')
        sys.exit()
