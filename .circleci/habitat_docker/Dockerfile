# syntax = docker/dockerfile:experimental
FROM nvidia/cudagl:10.1-base-ubuntu18.04 as base

ADD ./locobot_install_all.sh locobot_install_all.sh
RUN bash ./locobot_install_all.sh -t sim_only -p 3

RUN apt install -y psmisc curl

SHELL ["/bin/bash", "-c"]

#################
# Habitat install
#################
# last pyrobot commit that works with Habitat
RUN cd ~/low_cost_ws/src/pyrobot && git checkout 004cfd5bece0cec21bf216dffaa5733d0bc0a4af && \
    . /root/pyenv_pyrobot_python3/bin/activate && . /root/pyrobot_catkin_ws/devel/setup.bash && \
    pip install -e . && \
    pip install netifaces

# habitat system deps
RUN apt-get update || true
RUN apt-get install -y --no-install-recommends \
     libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev \
     patchelf


RUN mkdir habitat-test-scenes && cd habitat-test-scenes && wget http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip && \
    unzip habitat-test-scenes.zip && rm habitat-test-scenes.zip

RUN git clone --branch stable https://github.com/facebookresearch/habitat-sim.git --depth=1 && \
    cd habitat-sim && \
    . /root/pyenv_pyrobot_python3/bin/activate && . /root/pyrobot_catkin_ws/devel/setup.bash && \
    pip install -r requirements.txt && \
    python setup.py install --headless && \    
    cd .. && rm -rf habitat-sim

RUN patchelf --set-rpath '$ORIGIN' /root/pyenv_pyrobot_python3/lib/python3.6/site-packages/habitat_sim-0.1.4-py3.6-linux-x86_64.egg/habitat_sim/_ext/habitat_sim_bindings.cpython-36m-x86_64-linux-gnu.so
