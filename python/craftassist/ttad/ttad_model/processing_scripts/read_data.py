import json
from pprint import pprint
from random import choice

from models import *


def read_generations(f_name):
    res = []
    f = open(f_name)
    for line in f:
        if line[0] == "{":
            try:
                a_tree = ActionTree()
                a_tree_dict = json.loads(line.strip())
                a_desc = list(a_tree_dict.values())[0]["description"]
                del list(a_tree_dict.values())[0]["description"]
                a_tree.read_tree_dict(a_tree_dict)
                a_node = a_tree.root
                read_dict(a_node, a_tree)
                a_td = sorted(
                    write_top_down(a_node, len(a_desc.split())), key=lambda x: len(x["context"])
                )
                a_dfs = write_dfs(a_node)
                a_mrf = write_mrf(a_node)
                res += [
                    {
                        "action_description": a_desc,
                        "action_tree_dict": a_tree_dict,
                        "action_tree": a_tree,
                        "action_top_down": a_td,
                        "action_s2s_dfs": a_dfs,
                        "action_global_mrf": a_mrf,
                    }
                ]
            except Exception as e:
                print(e)
                print(line)
                break
    f.close()
    return res


generated_data = read_generations("example_trees.txt")

pprint(choice(generated_data), width=248)
