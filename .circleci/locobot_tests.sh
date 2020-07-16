#!/bin/bash
set -ex

source /opt/ros/melodic/setup.bash
export ORBSLAM2_LIBRARY_PATH=/root/low_cost_ws/src/pyrobot/robots/LoCoBot/install/../thirdparty/ORB_SLAM2
source /root/low_cost_ws/devel/setup.bash
source /root/pyenv_pyrobot_python3/bin/activate && source /root/pyrobot_catkin_ws/devel/setup.bash

pip install pyro4
roscore &

export PYRO_SERIALIZER='pickle'
export PYRO_SERIALIZERS_ACCEPTED='pickle'

LOCOBOT_IP=127.0.0.1 

python -m Pyro4.naming -n $LOCOBOT_IP &
sleep 10

python python/locobot/remote_locobot.py --ip $LOCOBOT_IP --backend habitat &
BGPID=$!
sleep 30
python python/locobot/test/smoke_test.py
kill -9 $BGPID
sleep 5

python python/locobot/remote_locobot.py --ip $LOCOBOT_IP --backend habitat &
BGPID=$!
sleep 30
pushd python/locobot/test
python test_habitat.py
popd
kill -9 $BGPID
sleep 5

python python/locobot/remote_locobot.py --ip $LOCOBOT_IP --backend habitat &
BGPID=$!
sleep 30
deactivate
source activate /root/miniconda3/envs/minecraft_env
python python/locobot/test/test_mover.py
kill -9 $BGPID
sleep 5

deactivate
source /root/pyenv_pyrobot_python3/bin/activate
python python/locobot/remote_locobot.py --ip $LOCOBOT_IP --backend habitat &
BGPID=$!
sleep 30
deactivate
source activate /root/miniconda3/envs/minecraft_env
mkdir -p models
curl http://craftassist.s3-us-west-2.amazonaws.com/pubr/models_folder.tar.gz -o crafassist_models.tar.gz 
tar -xzvf crafassist_models.tar.gz -C models/ --strip-components 1
curl http://craftassist.s3-us-west-2.amazonaws.com/pubr/ground_truth_data.txt -o ground_truth_data.txt
curl https://locobot-bucket.s3-us-west-2.amazonaws.com/perception_models.tar.gz -o locobot_models.tar.gz && \
tar -xzvf locobot_models.tar.gz -C models/
ls models
pip install facenet-pytorch
python python/locobot/test/test_perception_handlers.py
python python/locobot/test/test_locobot_agent.py
kill -9 $BGPID

python python/locobot/test/test_memory.py
