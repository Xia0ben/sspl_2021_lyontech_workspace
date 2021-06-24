FROM devrt/ros-devcontainer-vscode:melodic-desktop

USER root

# Most of the code is originated from:
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/10.0/ubuntu18.04-x86_64/base/Dockerfile

RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-10-0 \
    cuda-compat-10-0 \
    && ln -s cuda-10.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

######################## BEGIN: USER PACKAGES ##########################
RUN apt-get update && \
    apt-get install -y python-pip python-tk ros-$ROS_DISTRO-rospy-message-converter && \
    apt-get clean

RUN pip install scipy scikit-learn colour shapely aabbtree future matplotlib opencv-contrib-python==4.0.0.21
########################  END: USER PACKAGES  ##########################

# create workspace folder
RUN mkdir -p /workspace/src

# copy our algorithm to workspace folder
ADD . /workspace/src

# install dependencies defined in package.xml
RUN cd /workspace && /ros_entrypoint.sh rosdep install --from-paths src --ignore-src -r -y

# compile and install our algorithm
RUN cd /workspace && /ros_entrypoint.sh catkin_make install -DCMAKE_INSTALL_PREFIX=/opt/ros/$ROS_DISTRO

USER developer

# command to run the algorithm
CMD ["roslaunch", "wrs_challenge", "run.launch"]
