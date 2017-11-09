# Copyright (c) Facebook, Inc. and its affiliates.


# in acl branch
cd ~/Code/minecraft/python/craftassist/ttad/generation
python generate_actions.py -n 1000000 -chats dialogqe_cornellmov_persona.txt | awk 'NR % 3 == 2' > ~/Code/minecraft_acl/data/train_generated_1M_02_01.txt
python generate_actions.py -s 1212 -n 30000 -chats dialogqe_cornellmov_persona.txt | awk 'NR % 3 == 2' > ~/Code/minecraft_acl/data/valid_generated_30K_02_01.txt

cp ~/Code/minecraft/python/craftassist/ttad/validation_set/validation_set_rephrase_only.json ~/Code/minecraft_acl/data/valid_rephrase_02_01.txt

scp -r devfair_h2:/private/home/kavyasrinet/ttad_turk_data ~/Code/minecraft_acl/data/
