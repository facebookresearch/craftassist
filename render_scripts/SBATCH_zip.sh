#!/bin/bash
#SBATCH --job-name=minecraft_2D_render
#SBATCH --error=/private/home/zhuoyuan/minecraft_garage/error.out
#SBATCH --partition=learnfair
#SBATCH --comment="Monday Minecraft Eng"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --time=2000
#SBATCH --constraint pascal
zip -r gather.zip gather
