import os 
import sys

module_path = "../Modules/Models"

if module_path not in sys.path:
    sys.path.append(module_path)

from resnet import *