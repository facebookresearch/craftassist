"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

# TODO rewrite functions in intepreter and helpers as classes
# finer granularity of (code) objects
# interpreter is an input to interpret ref object, maybe clean that up?
class ReferenceObjectInterpreter:
    def __init__(self, interpret_reference_object):
        self.interpret_reference_object = interpret_reference_object

    def __call__(self, *args, **kwargs):
        return self.interpret_reference_object(*args, **kwargs)
