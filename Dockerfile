FROM ros:melodic-perception

SHELL [ "/bin/bash", "-c" ]

# install depending packages (install moveit! algorithms on the workspace side, since moveit-commander loads it from the workspace)
RUN apt-get update && \
    apt-get install -y git ros-$ROS_DISTRO-moveit ros-$ROS_DISTRO-moveit-commander ros-$ROS_DISTRO-move-base-msgs ros-$ROS_DISTRO-ros-numpy ros-$ROS_DISTRO-geometry && \
    apt-get clean

# install bio_ik
RUN source /opt/ros/$ROS_DISTRO/setup.bash && \
    mkdir -p /bio_ik_ws/src && \
    cd /bio_ik_ws/src && \
    catkin_init_workspace && \
    git clone --depth=1 https://github.com/TAMS-Group/bio_ik.git && \
    cd .. && \
    catkin_make install -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/ros/$ROS_DISTRO -DCATKIN_ENABLE_TESTING=0 && \
    cd / && rm -r /bio_ik_ws

######################## BEGIN: NVIDIA ##########################

# Most of the code is originated from:
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/10.0/ubuntu18.04-x86_64/base/Dockerfile

# RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
#     echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
#     echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
#
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     cuda-cudart-10-0 \
#     cuda-compat-10-0 \
#     && ln -s cuda-10.0 /usr/local/cuda && \
#     rm -rf /var/lib/apt/lists/*
#
# RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
#     echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
#
# ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
#
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
########################  END: NVIDIA  ##########################

######################## BEGIN: USER PACKAGES ##########################
RUN apt-get update && \
    apt-get install -y python-pip ros-$ROS_DISTRO-rospy-message-converter && \
    apt-get clean

RUN pip install scipy scikit-learn colour shapely aabbtree

# RUN apt-get update && \
#     apt-get install -y python3-pip && \
#     apt-get clean

# RUN python3 -m pip install --upgrade pip

# RUN pip3 install rospkg numpy opencv-python requests cython yolo34py
########################  END: USER PACKAGES  ##########################

# create workspace folder
RUN mkdir -p /workspace/src

# copy our algorithm to workspace folder
ADD . /workspace/src

# install dependencies defined in package.xml
RUN cd /workspace && /ros_entrypoint.sh rosdep install --from-paths src --ignore-src -r -y

# compile and install our algorithm
RUN cd /workspace && /ros_entrypoint.sh catkin_make install -DCMAKE_INSTALL_PREFIX=/opt/ros/$ROS_DISTRO

# command to run the algorithm
CMD roslaunch wrs_challenge run.launch
