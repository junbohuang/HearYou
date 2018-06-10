import os
import shutil
import sys
import pandas as pd
import numpy as np

from infrastructure.helper_function import *


def main():
    if not os.path.exists("./data/iemocap/wav_train/") and \
            os.path.exists("./data/iemocap/wav_valid/") and \
            os.path.exists("./data/iemocap/wav_test/"):
        split(train_perc=0.85, val_perc=0.10)
    else:
        pass


split(train_perc=0.85, val_perc=0.10)






