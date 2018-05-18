import os
import sys
import shutil
import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('agg') # uncomment this in grid. 
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

def plot_emo_num(emo_num, server="local"):
    """plot distribution of emotion categoriy.
    
    Args:
    emo_num(list): list of unique emotion
    server(str): local - plot emotion distribution; ssh - without plotting. 
    
    Return:
    save (or plot) emotion categoriy plot.
    """
    
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
    fig.tight_layout()
    fig.savefig("./figures/emotion_distribution.png")
    if server =="local":
        plt.show()
    elif server =="ssh":
        pass
    else: 
        raise ValueError("server not defined.")    

def get_emotion_from_filename(df, filename):
    emotion = df.loc[df['name'] == filename, 'emotion'].to_string(index = False)

    return emotion

def check_emotion(prediction, label):
    result = 0
    if label == prediction:
        result = 1
    else: pass
    return result

def split(train_perc = 0.85, val_perc = 0.10, test_perc = 0.05):
    emo_category = "./data/iemocap/emotions/"
    emo_used = ['fru','neu','ang','exc','sad']

    train = []
    val = []
    test = []

    for i in range(len(emo_used)):
        wav_list = os.listdir(emo_category + emo_used[i])
        p = np.random.permutation(len(wav_list))

        train_onset = 0
        train_offset = int(np.ceil(train_perc * len(p)))
        val_onset = int(np.ceil(train_perc * len(p)))
        val_offset = int(np.ceil((train_perc + val_perc) * len(p)))
        test_onset = int(np.ceil((train_perc + val_perc) * len(p)))
        test_offset = int(len(p)-1)

        if not os.path.exists("./data/iemocap/wav_train/" + emo_used[i]):
            os.makedirs("./data/iemocap/wav_train/" + emo_used[i])
        else: pass
        for n in p[train_onset:train_offset]:
            train.append(wav_list[n])
            src = "./data/iemocap/emotions/" + emo_used[i] + "/" + wav_list[n]
            dst = "./data/iemocap/wav_train/" + emo_used[i]
            shutil.copy(src, dst)

        if not os.path.exists("./data/iemocap/wav_valid/" + emo_used[i]):
            os.makedirs("./data/iemocap/wav_valid/" + emo_used[i])
        else: pass
        for n in p[val_onset:val_offset]:
            val.append(wav_list[n])
            src = "./data/iemocap/emotions/" + emo_used[i] + "/" + wav_list[n]
            dst = "./data/iemocap/wav_valid/" + emo_used[i]
            shutil.copy(src, dst)

        if not os.path.exists("./data/iemocap/wav_test/"):
            os.makedirs("./data/iemocap/wav_test/")
        else: pass
        for n in p[test_onset:test_offset]:
            test.append(wav_list[n])
            src = "./data/iemocap/emotions/" + emo_used[i] + "/" + wav_list[n]
            dst = "./data/iemocap/wav_test/"
            shutil.copy(src, dst)


    print("number of training files:",len(train))
    print("number of validation files:",len(val))
    print("number of testing files:",len(test))

def get_accuracy():
    emotion_data = pd.read_csv("emotion_data.csv", header=0, index_col=0)
    col_name = ["filename", "prediction"]
    results = pd.read_csv("Results.txt", delimiter=".wav  ", names=col_name, engine='python')
    filenames = results['filename']
    predictions = results['prediction']

    correct = 0
    count = 0
    accuracy = 0
    for i in range(len(filenames)):
        filename = filenames[i]
        prediction = predictions[i]
        label = get_emotion_from_filename(emotion_data, filename)

        correct += check_emotion(prediction, label)
        count += 1
    accuracy = correct / count

    return print("accuracy:", accuracy, "\nwhere correct prediciton count is:", correct, "out of", count, "predictions")


