### Parser Training Instructions

First, we need to pre-generate some templated data to train the model on. 500K examples should be a good start:
```
$ cd ~/minecraft/python/craftassist/ttad/generation_dialogues
$ python generate_dialogue.py -n 500000 > ../data/ttad_model/generated_dialogues.txt
```

This generates a text file. We next pre-process the data into the json format required by the training script
```
$ cd ../ttad_model/
$ python make_dataset.py ../data/ttad_model/generated_dialogues.txt ../data/ttad_model/dialogue_test_data.json
```

Now we run a script to build the grammar automatically from the generated examples:
```
$ python make_action_grammar.py ../data/ttad_model/dialogue_test_data.json ../data/ttad_model/dialogue_test_grammar.json
```

We are now ready to train the model with:
```
$ python train_model.py -cuda -rp 0 -rsm none -rst templated -df ../data/ttad_model/dialogue_test_data.json -atf ../data/ttad_model/dialogue_test_grammar.json -mn ../data/ttad_model/models/dialogue_test_model
```

The -rp, -rsm and -rst options are required to train on only templated data, but feel free to experiment with the model parameters. Once you're done, choose which epoch you want the parameters for, e.g. if 58 had the best validation accuracy:
```
$ cp ../data/ttad_model/models/dialogue_test_model_58.pth ../data/ttad_model/models/dialogue_test_model.pth
```

You can now use that model. In a Python terminal from the ttad_models directory, run:
```
from pprint import pprint
from ttad_model_wrapper import *
from random import choice

ttad = ActionDictBuilder('../data/ttad_model/models/dialogue_test_model', action_tree_path='../data/ttad_model/dialogue_test_grammar.json')

data = json.load(open('../data/ttad_model/dialogue_test_data.json'))

exple = choice(data['test']['templated'])
pprint((exple, ttad.parse(exple[0])))
```
