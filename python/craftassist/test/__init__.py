import os
import sys

dir_test = os.path.dirname(__file__)
dir_craftassist = os.path.join(dir_test, "..")

sys.path.append(dir_craftassist)
sys.path.insert(0, dir_test)  # insert 0 so that Agent is pulled from here
