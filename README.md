The goal of this project is to build an intelligent, collaborative assistant bot in the game of [Minecraft](https://www.minecraft.net/en-us/)<sup>1</sup> that can perform a wide variety of tasks specified by human players. Its primary purpose is to be a tool for artifical intelligence researchers interested in grounded dialogue and interactive learning. This project is in active development.

A detailed outline and documentation is available in [this paper](https://arxiv.org/abs/1907.08584)

This release is motivated by a long-term research agenda described [here](https://research.fb.com/publications/why-build-an-assistant-in-minecraft/).

![GIF of Gameplay With Bot](https://craftassist.s3-us-west-2.amazonaws.com/pubr/bot_46.gif)

## Installation & Getting Started

Do this section *before* cloning the repo.

### Dependencies

Make sure the following packages have already been installed before moving on:
* CMake
* Python3
* Glog
* Boost
* Eigen
* gcc version: 7.4.0 on ubuntu 18.04
* For Mac users:
  * LLVM version < 10 to successfully use clang. [Working with multiple versions of Xcode](https://medium.com/@hacknicity/working-with-multiple-versions-of-xcode-e331c01aa6bc).

### Install git-lfs

```
# OSX
brew install git-lfs
git lfs install

# On Ubuntu
sudo add-apt-repository ppa:git-core/ppa
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```

### Checking out the repo

Use this command, or your submodules will not be pulled, and your clone will take a very long time:

```
git lfs clone --recursive git@github.com:facebookresearch/craftassist.git
```

Now `cd craftassist` and copy the large data file and untar it to the correct directories:

```
curl http://craftassist.s3-us-west-2.amazonaws.com/pubr/models_folder.tar.gz -o models_folder.tar.gz
tar -xzvf models_folder.tar.gz -C python/craftassist/models/ --strip-components 1
curl http://craftassist.s3-us-west-2.amazonaws.com/pubr/ground_truth_data.txt -o python/craftassist/ground_truth_data.txt

```

### Python Requirements: Using A Conda Environment
To build a conda environment that supports this release:
```
# Create a new env preloaded with the conda install dependencies
conda create -n minecraft_env python==3.7.4 pip numpy scikit-learn==0.19.1 pytorch torchvision -c conda-forge -c pytorch
conda activate minecraft_env

# Install all of the rest of the dependencies with pip
pip install -r requirements.txt
```
Then activate this environment whenever you want to run the agent.

### Building client and server

To build Cuberite and the C++ Minecraft client:
```
make
```

### Run the Cuberite instance

Run the following command

```
python ./python/cuberite_process.py
```
to start an instance of cuberite instance listening on `localhost:25565`


## Connecting your Minecraft game client (so you can see what's happening)

Buy and download the [official Minecraft client](https://my.minecraft.net/en-us/store/minecraft/).

You can inspect the world and view the Minecraft agent's actions by logging into the
running Cuberite instance from the game client.

To connect the client to the running Cuberite instance, click in the Minecraft client:

```
Multiplayer > Direct Connect > localhost:25565
```

#### Error: Unsupported Protocol Version

Minecraft has recently release v1.15.2, and our Cuberite system supports at most v1.12

[Please follow these instructions](https://help.minecraft.net/hc/en-us/articles/360034754852-Changing-game-versions-) to add a 1.12.x profile and use it to connect.

## Running the interactive V0 agent

Assuming you have set up the [Cuberite server](https://github.com/facebookresearch/craftassist#run-the-cuberite-instance)
and the [client](https://github.com/facebookresearch/craftassist#connecting-your-minecraft-game-client-so-you-can-see-whats-happening), in a separate tab, run:

```
python ./python/craftassist/craftassist_agent.py
```

You should see a new bot player join the game.

Chat with the bot by pressing `t` to open the dialogue box, and `Enter` to submit.

Use the `w`, `a`, `s`, and `d` keys to navigate, left and right mouse clicks to destroy and place blocks, and `e` to open your inventory and select blocks to place.

## Running tests

```
./python/craftassist/test.sh
```

## Datasets

Download links to the datasets described in section 6 of [Technical Whitepaper](https://arxiv.org/abs/1907.08584) are provided here:

- **The house dataset**: https://craftassist.s3-us-west-2.amazonaws.com/pubr/house_data.tar.gz
- **The segmentation dataset**: https://craftassist.s3-us-west-2.amazonaws.com/pubr/instance_segmentation_data.tar.gz
- **The dialogue dataset**: https://craftassist.s3-us-west-2.amazonaws.com/pubr/dialogue_data.tar.gz

In the root of each tarball is a README that details the file structure contained within.

## Citation

If you would like to cite this repository in your research, please cite [the CraftAssist paper](https://arxiv.org/abs/1907.08584).

```
@misc{gray2019craftassist,
    title={CraftAssist: A Framework for Dialogue-enabled Interactive Agents},
    author={Jonathan Gray and Kavya Srinet and Yacine Jernite and Haonan Yu and Zhuoyuan Chen and Demi Guo and Siddharth Goyal and C. Lawrence Zitnick and Arthur Szlam},
    year={2019},
    eprint={1907.08584},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```

## License

CraftAssist is [MIT licensed](./LICENSE).

<sup>1</sup> Minecraft features: © Mojang Synergies AB included courtesy of Mojang AB
