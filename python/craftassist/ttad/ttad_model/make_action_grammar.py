import json
import sys

from pprint import pprint
from acl_models import *

data_file = sys.argv[1]
grammar_file = sys.argv[2]

print("loading", data_file)
data_dct = json.load(open(data_file))
print("loaded data")

a_tree = ActionTree()
for spl, spl_dct in data_dct.items():
    for d_type, ls in spl_dct.items():
        print("reading", spl, d_type)
        a_tree.build_from_list([t for d, t in ls])


a_tree_dct = a_tree.to_dict()
json.dump(a_tree_dct, open(grammar_file, "w"))

pprint(a_tree_dct)
