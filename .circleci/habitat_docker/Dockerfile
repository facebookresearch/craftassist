# syntax = docker/dockerfile:experimental
FROM nvidia/cudagl:10.1-base-ubuntu18.04 as base

ADD ./locobot_install_all.sh locobot_install_all.sh
RUN bash ./locobot_install_all.sh -t sim_only -p 3

RUN apt install -y psmisc curl

SHELL ["/bin/bash", "-c"]

#################
# Habitat install
#################

# habitat system deps
RUN apt-get update || true
RUN apt-get install -y --no-install-recommends \
     libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev \
     patchelf libbullet-dev


RUN mkdir habitat-test-scenes && cd habitat-test-scenes && wget http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip && \
    unzip habitat-test-scenes.zip && rm habitat-test-scenes.zip

# this was locally put in the directory creating the docker image with select environments from
# the replica dataset: https://github.com/facebookresearch/Replica-Dataset/blob/master/download.sh
ADD ./Replica-Dataset /Replica-Dataset

ADD ./specular.patch specular.patch

RUN git clone https://github.com/facebookresearch/habitat-sim && \
    cd habitat-sim && \
    git checkout eed76b2f3419f3cfbcc6489aa0fe1b69cf89237a && \
    patch -p1 <../specular.patch && \
    . /root/pyenv_pyrobot_python3/bin/activate && . /root/pyrobot_catkin_ws/devel/setup.bash && \
    pip install -r requirements.txt && \
    python setup.py install --headless --with-bullet && \    
    cd .. && rm -rf habitat-sim && \
    git clone https://github.com/facebookresearch/habitat-api && \
    cd habitat-api && \
    git checkout 01696bb10d5735b253691b764286562af87fdbc3 && \
    pip install -e .

RUN patchelf --set-rpath '$ORIGIN' /root/pyenv_pyrobot_python3/lib/python3.6/site-packages/habitat_sim-0.1.4-py3.6-linux-x86_64.egg/habitat_sim/_ext/habitat_sim_bindings.cpython-36m-x86_64-linux-gnu.so

# tested pyrobot commit that works with Habitat
RUN cd ~/low_cost_ws/src/pyrobot && git checkout Develop && \
    git pull https://github.com/facebookresearch/pyrobot Develop && \
    git checkout 94fc6d6d177f5ac737e82b04844ff792d94b0dc5 && \
    . /root/pyenv_pyrobot_python3/bin/activate && . /root/pyrobot_catkin_ws/devel/setup.bash && \
    pip install -e . && \
    pip install netifaces pyro4