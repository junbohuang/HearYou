import os
import sys
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt

from infrastructure.helper_function import *

emo_dir = "./data/iemocap/emotions"
emotions = [emotion for emotion in os.listdir(emo_dir) if not emotion.startswith('.')]
emo_num = {}
for emotion in emotions:
    emo_num[emotion] = len(os.listdir(emo_dir + "/" + emotion))
p = plot_emo_num(emo_num)
