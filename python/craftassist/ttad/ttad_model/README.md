### Parser Training Instructions

First, we need to pre-generate some templated data to train the model on. 500K examples should be a good start:
```
$ cd ~/minecraft/python/craftassist/ttad/generation_dialogues
$ python generate_dialogue.py -n 500000 > ../ttad_model/data/generated_dialogues.txt
```

This generates a text file. We next pre-process the data into the json format required by the training script
```
$ cd ../ttad_model/
$ python make_dataset.py data/generated_dialogues.txt acl_data/dialogue_data.json
```

Now we run a script to build the grammar automatically from the generated examples:
```
python make_action_grammar.py acl_data/dialogue_data.json acl_data/dialogue_grammar.json
```

We are now ready to train the model with:
```
python train_model.py -cuda -rp 0 -rsm none -rst templated -df acl_data/dialogue_data.json -atf acl_data/dialogue_grammar.json -mn data/models/dialogue_test_model
```

The -rp, -rsm and -rst options are required to train on only templated data, but feel free to experiment with the model parameters. Once you're done, choose which epoch you want the parameters for, e.g. if 58 had the best validation accuracy:
```
cp data/models/dialogue_test_model_58.pth data/models/dialogue_test_model.pth
```

You can now use that model. In a Python terminal from the ttad_models directory, run:
```
from pprint import pprint
from ttad_model_wrapper import *
from random import choice

ttad = ActionDictBuilder('data/models/dialogue_test_model', action_tree_path='acl_data/dialogue_grammar.json')

data = json.load(open('acl_data/dialogue_data.json'))

exple = choice(data['test']['templated'])
pprint((exple, ttad.parse(exple[0])))
```
