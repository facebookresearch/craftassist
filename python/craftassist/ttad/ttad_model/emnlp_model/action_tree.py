"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

# define the grammar

from collections import OrderedDict


#####
## define node types
class SpanSingleLeaf:
    def __init__(self, node_id, name):
        self.node_type = "span-single"
        self.name = name
        self.node_id = node_id


def is_span(val):
    try:
        # previous version
        # (b, c) = val
        # return all([type(v) == int for v in [b, c]])
        a, (b, c) = val
        return all([type(v) == int for v in [a, b, c]])
    except (ValueError, TypeError):
        return False


def is_span_dialogue(val):
    try:
        a, (b, c) = val
        return all([type(v) == int for v in [a, b, c]])
    except (ValueError, TypeError):
        return False


class SpanSetLeaf:
    def __init__(self, node_id, name):
        self.node_type = "span-set"
        self.name = name
        self.node_id = node_id


def is_span_list(val):
    res = type(val) == list and len(val) > 0 and all([is_span(v) for v in val])
    return res


class CategoricalSingleLeaf:
    def __init__(self, node_id, name, choices=None):
        self.node_type = "categorical-single"
        self.name = name
        self.node_id = node_id
        self.choices = [] if choices is None else choices[:]


def is_cat(val):
    return type(val) == str or val is True or val is False


class CategoricalSetLeaf:
    def __init__(self, node_id, name, choices=None):
        self.node_type = "categorical-set"
        self.name = name
        self.node_id = node_id
        self.choices = [] if choices is None else choices[:]


def is_cat_list(val):
    res = type(val) == list and len(val) > 0 and all([is_cat(v) for v in val])
    return res


def make_leaf(node_id, name, value=None, node_type=""):
    if (
        type(value) == list
        and len(value) == 2
        and (type(value[0]) == int and type(value[1]) == int)
    ):
        value = [0, value]
    if node_type == "span-single" or is_span(value):
        return SpanSingleLeaf(node_id, name)
    elif node_type == "span-set" or is_span_list(value):
        return SpanSetLeaf(node_id, name)
    elif node_type == "categorical-single" or is_cat(value):
        return CategoricalSingleLeaf(node_id, name)
    elif node_type == "categorical-set" or is_cat_list(value):
        return CategoricalSetLeaf(node_id, name)
    else:
        print("M", value)
        raise NotImplementedError


class InternalNode:
    def __init__(self, node_id, name):
        self.node_type = "internal"
        self.node_id = node_id
        self.name = name
        self.internal_children = OrderedDict()
        self.leaf_children = OrderedDict()
        self.children = OrderedDict()

    def add_int_node(self, name):
        new_node = InternalNode(self.node_id + "-" + name, name)
        node = self.internal_children.get(name, new_node)
        self.internal_children[name] = node
        self.children[name] = node
        return node

    def add_leaf_node(self, name, value):
        node_id = self.node_id + ":" + name
        new_node = make_leaf(node_id, name, value=value)
        node = self.leaf_children.get(name, new_node)
        if is_cat(value) and value not in node.choices:
            node.choices += [value]
        elif is_cat_list(value):
            for v in value:
                if v not in node.choices:
                    node.choices += [v]
        self.leaf_children[name] = node
        self.children[name] = node
        return node


def is_sub_tree(val):
    return type(val) == dict


####
# make full tree
class ActionTree:
    def __init__(self):
        self.root = InternalNode("root", "root")

    def _add_subtree_dict(self, node, sub_tree):
        for k, val in sub_tree.items():
            if is_sub_tree(val):
                child_node = node.add_int_node(k)
                self._add_subtree_dict(child_node, val)
            else:
                if k == "location":
                    print(sub_tree)
                    raise NotImplementedError
                node.add_leaf_node(k, val)

    def build_from_list(self, tree_list):
        for i, parse_tree in enumerate(tree_list):
            self._add_subtree_dict(self.root, parse_tree)

    def _sub_to_dict(self, node):
        res = OrderedDict()
        for k, val in node.internal_children.items():
            res[k] = (val.node_type, val.node_id, self._sub_to_dict(val))
        for k, val in node.leaf_children.items():
            res[k] = (
                val.node_type,
                val.node_id,
                val.choices if "categorical" in val.node_type else "",
            )
        return res

    def to_dict(self):
        return self._sub_to_dict(self.root)

    def _sub_from_dict(self, node, sub_tree):
        for name, (node_type, node_id, value) in sub_tree.items():
            if node_type == "internal":
                child_node = node.add_int_node(name)
                assert node_id == child_node.node_id
                self._sub_from_dict(child_node, value)
            else:
                new_node = make_leaf(node_id, name, node_type=node_type)
                if "categorical" in node_type:
                    new_node.choices = value
                node.leaf_children[name] = new_node
                node.children[name] = new_node

    def from_dict(self, a_tree):
        self._sub_from_dict(self.root, a_tree)
