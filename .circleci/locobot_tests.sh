#!/bin/bash
set -ex

source /opt/ros/melodic/setup.bash
export ORBSLAM2_LIBRARY_PATH=/root/low_cost_ws/src/pyrobot/robots/LoCoBot/install/../thirdparty/ORB_SLAM2
source /root/low_cost_ws/devel/setup.bash
source /root/pyenv_pyrobot_python3/bin/activate && source /root/pyrobot_catkin_ws/devel/setup.bash

pip install pyro4
pip install -r requirements.txt
roscore &

export PYRO_SERIALIZER='pickle'
export PYRO_SERIALIZERS_ACCEPTED='pickle'

LOCOBOT_IP=127.0.0.1 
SHARED_PATH=/shared
pip install pytest
pip install pytest-cov

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
pytest --cov-report=xml:$SHARED_PATH/test_habitat.xml --cov=../ test_habitat.py --disable-pytest-warnings
popd
kill -9 $BGPID
sleep 5

COV_RELATIVE=python/locobot
python python/locobot/remote_locobot.py --ip $LOCOBOT_IP --backend habitat &
BGPID=$!
sleep 30
deactivate
source activate /root/miniconda3/envs/minecraft_env
pip install pytest
pip install pytest-cov
pytest --cov-report=xml:$SHARED_PATH/test_mover.xml --cov=$COV_RELATIVE python/locobot/test/test_mover.py --disable-pytest-warnings
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
mkdir -p datasets
curl http://craftassist.s3-us-west-2.amazonaws.com/pubr/datasets_folder.tar.gz -o crafassist_datasets.tar.gz 
tar -xzvf crafassist_datasets.tar.gz -C datasets/ --strip-components 1
curl https://locobot-bucket.s3-us-west-2.amazonaws.com/perception_models.tar.gz -o locobot_models.tar.gz && \
tar -xzvf locobot_models.tar.gz -C models/
curl https://locobot-bucket.s3-us-west-2.amazonaws.com/perception_test_assets.tar.gz -o perception_test_assets.tar.gz
tar -xzvf perception_test_assets.tar.gz -C python/locobot/test/test_assets/
ls models
pip install facenet-pytorch
pip install -r requirements.txt
pytest --cov-report=xml:$SHARED_PATH/test_locobot_agent.xml --cov=$COV_RELATIVE python/locobot/test/test_locobot_agent.py --disable-pytest-warnings
pytest --cov-report=xml:$SHARED_PATH/test_perception_handlers.xml --cov=$COV_RELATIVE python/locobot/test/test_perception_handlers.py --disable-pytest-warnings

kill -9 $BGPID
pytest --cov-report=xml:$SHARED_PATH/test_memory.xml --cov=$COV_RELATIVE python/locobot/test/test_memory.py --disable-pytest-warnings

