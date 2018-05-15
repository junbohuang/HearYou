import os
import sys
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def get_wav_path(filename, session, cwd):
    """get wav file path based on a filename (emo_data['name'].values[n])
    
    Args:
    filenames(str): a wav filename from pandas dataframe.
    cwd(str): current working directory.
    
    Return: 
    wav_path(str): wav file path with respect to filename.
    """
    
    setting = filename[0:-5]
    wav_path = str(cwd) + "/data/iemocap/IEMOCAP_full_release/" + session + \
        "/sentences/wav/" + setting 
    
    return wav_path


def setup_folder_structure(emotions, cwd):
    if not os.path.exists("./figures"):
        os.makedirs("./figures")
    else: pass
    if not os.path.exists(cwd + "/data/iemocap/emotions"):
        os.makedirs(cwd + "/data/iemocap/emotions")
    else: pass
    
    emo_root = cwd + "/data/iemocap/emotions"
    for emotion in emotions:
        if not os.path.exists(emo_root + "/" + emotion):
            os.makedirs(emo_root + "/" + emotion)
        else: pass
    
    return emo_root

def get_dict(emo_data):
    emo_dict = dict(zip(emo_data["name"].values, emo_data["emotion"].values))
    return emo_dict

def move_wav(wav_path, emo_root, filename, emotion):
    src = wav_path + "/" + filename + ".wav"
    dst = emo_root + "/" + emotion
    shutil.copy(src, dst)
    return 

def plot_emo_num(emo_num):
    
    fig, ax = plt.subplots(figsize=(10,7))

    index = np.arange(len(emo_num))
    bar_width = 0.5

    rects = ax.bar(index, list(emo_num.values()), bar_width)

    ax.set_xlabel('Emotion')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Emotions')
    ax.set_xticks(index)
    ax.set_xticklabels(list(emo_num.keys()))
    ax.legend()
    labels = list(emo_num.values())

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, label,
                ha='center', va='bottom')
    fig.savefig("./figures/emotion_distribution.png")
    # comment below in the Grid
    # plt.show()
    
    
