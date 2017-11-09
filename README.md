Continuous integration status (master)

[![CircleCI](https://circleci.com/gh/fairinternal/minecraft.svg?style=svg&circle-token=26864dc39f670a5bc819dab48d677556db031df6)](https://circleci.com/gh/fairinternal/minecraft)

# Getting Started

## The Project
The goal of this project is to build an intelligent in-game assistant. We call this an `agent`. The
agent is instantiated as another player in the game. At the moment, the agent can:
- Build things
- Make copies
- Destroy block objects
- Move around
- Revert what it did
- Remember names/ tags of things when you tell it
- Answer questions about objects in the environment
- Stop when you ask it
- Resume a paused task
- Dig
- Fill negative shapes
- Spawn mobs
- Do a dance
- Freebuild (complete something half-built by a human player)

The agent's primary interface is via natural language using Minecraft chat. We aim to eventually make an agent that is collaborative, and that Minecraft players might find fun to play with.

## Prerequisites

Do this section *before* cloning the repo.

### Dependencies
Make sure the following packages have already been installed before moving on:
* CMake
* Python3
* Glog
* Boost
* Eigen
* For Mac users:
  * LLVM version < 10 to successfully use clang. [Working with multiple versions of Xcode](https://medium.com/@hacknicity/working-with-multiple-versions-of-xcode-e331c01aa6bc).
* For FAIR cluster users:
  * Environment [Modules](http://modules.sourceforge.net/) is needed for `module load`.

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

## Checking out the repo

Use this command, or your submodules will not be pulled, and your clone will take a very long time:

```
git lfs clone --recursive git@github.com:fairinternal/minecraft.git
```

Now `cd minecraft` before proceeding to the following sections..


## Building client and server

To build Cuberite and the C++ Minecraft client on the *FAIR cluster* :
```
module load anaconda3/5.0.1
source activate /private/home/kavyasrinet/.conda/envs/minecraft_env
make
```

The first two lines are *only* needed if you are building on the FAIR cluster.

## Run the Cuberite instance

Run the following command

```
python ./python/cuberite_process.py
```
to start an instance of cuberite on your local machine/ FAIR cluster.


## Connecting your client (so you can see what's happening)

You can inspect the world and view the Minecraft agent's actions by logging into the
running Cuberite instance from the official Minecraft client.

If Cuberite is running remotely on a FAIR devserver, create an ssh tunnel from
your Macbook:

```
ssh -fN -L 25565:100.97.69.35:25565 -J prn-fairjmp03 kavyasrinet@100.97.69.35
```
Be sure to replace `100.97.69.35` with your devserver's IP and `kavyasrinet` with
your username.

To connect the client to the running Cuberite instance:

In the Minecraft client, click :
```
Multiplayer > Direct Connect > localhost:25565
```

### Error: Unsupported Protocol Version

Minecraft has recently release v1.13, and our Cuberite system supports at most v1.12

[Please follow these instructions](https://help.mojang.com/customer/portal/articles/1475923-changing-game-versions) to add a 1.12.x profile and use it to connect.

## Running the interactive V0 agent

Assuming you have set up the [Cuberite server](https://github.com/fairinternal/minecraft#run-the-cuberite-instance)
and the [client](https://github.com/fairinternal/minecraft#visualization), in a separate tab, run:

```
python ./python/craftassist/craftassist_agent.py
```

Now you should be able to see a `bot.x` in your Minecraft client. `x` being the current timestamp.

You can chat with the bot by pressing `t` on your keyboard. It will open up a dialog box for your text.
The client screen will also show the agent's response, if any.

You can use the keys: `w`, `a`, `s` and `d` to navigate to the front, left, back and right respectively,
in the environment.

## The semantic parsing dataset

To generate the semantic parsing dataset (natural language utterances and their respective action dictionaries), run:
```
python ./python/craftassist/ttad/generation_dialogues/generate_dialogue.py -n N --action_type ACTION_TYPE
```
where `N` is the number of utterances in the dataset.
and `ACTION_TYPE` is one of : ``` move / build / destroy / noop / stop/ resume / dig / copy / undo / fill / spawn / freebuild / dance / get_memory / put_memory```

By default, the script produces 100 utterances of a mixture of all the actions mentioned above.

## Papers

Here is a link to our paper describing the project : [The Minecraft Project]()

Here is a link to the paper describing the [semantic parsing dataset](https://arxiv.org/pdf/1905.01978.pdf)
