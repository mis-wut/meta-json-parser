import sys
import pprint

import json

class Visitor(object):
    def __init__(self):
        self.level = 0
        self.keys = {}

    def node_type(self, node):
        if not isinstance(node, dict):
            raise NameError("Node is not dict " + node)

        if "type" not in node:
            raise NameError("Type not in node")
        return node["type"]

    def visit_object(self, node):
        default_mapping = {
            "boolean" : "JBool<uint8_t, {name}>",
            "integer" : "JNumber<uint32_t, {name}>",
            "number" : "JRealNumber<float, {name}>",
            "string" : "JStringVariant<{name}, 32>",
            "null" : "JStringVariant<{name}, 32>",
        }

        prefix = "    " * (self.level)
        if self.level == 0:
            ret = "// DICT\n"
            ret += "template<template<class, int> class StringFun, class DictOpts>"
            ret += "\nusing DictCreator = JDict < mp_list <\n"
        else:
            ret = prefix + "JDict < mp_list <\n"

        self.level += 1

        properties = node["properties"]
        if "metaparser" in node:
            default_mapping.update(node["metaparser"])

        for key, value in properties.items():
            cuda_key = f"K_L{self.level}_{key}"
            value_type = self.visit(value)
            if value["type"] == "object": # TODO: array
                ret += f"mp_list < {cuda_key},\n" + value_type
            else:

                if "transformation" in value:
                    transformation = value["transformation"]
                else: # only now check default
                    transformation = default_mapping[value_type]

                transformation = transformation.format(name=cuda_key)

                ret += prefix + f"mp_list<{cuda_key}, {transformation}>,\n"

            self.keys[cuda_key] = key
        if properties.items(): # we did have some properites 
            ret = ret[:-2] # remove last ,

        self.level -= 1
        if self.level > 0:
            ret += prefix + "\n>, DictOpts>\n>\n>"
        else:
            ret += ">,\nDictOpts>;"

        return ret

    def visit_array(self, node):
        ret = "// Array\n"
        self.level += 1
        aitems = node["items"]

        # TODO: check what if more then one type of objects in array
        ret += " " * self.level + self.visit(aitems)
        self.level -= 1
        return ret

    def visit_string(self, node):
        return self.node_type(node)

    def visit_integer(self, node):
        return self.node_type(node)

    def visit_number(self, node):
        return self.node_type(node)

    def visit_null(self, node):
        return self.node_type(node)

    def visit_boolean(self, node):
        return self.node_type(node)

    def visit(self, node):
        strategy = self.node_type(node)
        meth = getattr(self, 'visit_' + strategy, None)
        if meth:
            return  meth(node)
        raise NameError("No visitor found for " + strategy)

    def generate_keys(self):
        ret = "\n// KEYS\n"
        for key, value in self.keys.items():
            key_sting  = f'"{value}"'
            ret += f"using {key} = metastring({key_sting});\n"
        return ret

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"run {sys.argv[0]} file.json")
        exit()

    with open(sys.argv[1], 'r') as f:
        node = json.load(f)

    vis = Visitor()
    out = vis.visit(node)
    print(vis.generate_keys())
    print(out)
