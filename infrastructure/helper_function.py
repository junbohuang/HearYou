import os
import itertools
import shutil
import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('agg') # uncomment this in grid. 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from config import *


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

def plot_emo_num(emo_num, server = "local"):
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

def split(train_perc = 0.85, val_perc = 0.10):
    emo_category = "./data/iemocap/emotions/"
    emo_used = config.emotions
    print("emotion:", emo_used)
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
    y_true = []
    for i in range(len(filenames)):
        filename = filenames[i]
        prediction = predictions[i]
        label = get_emotion_from_filename(emotion_data, filename)
        y_true.append(label)

        correct += check_emotion(prediction, label)
        count += 1
    accuracy = correct / count

    # confusion matrix
    cnf_matrix = confusion_matrix(y_true, predictions, labels = config.emotions)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes= config.emotions,
                          title='Confusion matrix, without normalization')
    plt.savefig("./figures/confusion_matrix_non_normalized.png")

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=config.emotions, normalize=True,
                        title='Normalized confusion matrix')
    plt.savefig("./figures/confusion_matrix_normalized.png")
    #plt.show()

    # classification_report
    print(classification_report(y_true, predictions, target_names=config.emotions))
    return print("accuracy:", accuracy, "\nwhere correct prediciton count is:", correct, "out of", count, "predictions")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_csv():
    cwd = os.getcwd()
    sessions = [session for session in os.listdir(cwd + "/data/iemocap/IEMOCAP_full_release/")
                if session.startswith('Session')]

    emotion_data = pd.DataFrame(columns=["name", "emotion"])

    for session in sessions:
        path_txt = cwd + "/data/iemocap/IEMOCAP_full_release/" + session + "/dialog/EmoEvaluation"
        path_wav = cwd + "/data/iemocap/IEMOCAP_full_release/" + session + "/sentences/wav"
        filenames_txt = [txt for txt in os.listdir(path_txt) if txt.startswith('Ses') and txt.endswith('.txt')]
        filenames_wav = [wav for wav in os.listdir(path_wav) if wav.startswith('Ses') and wav.endswith('.wav')]
        aux_data = pd.DataFrame(columns=["time", "name", "emotion", "vad"])

        for name in filenames_txt:
            aux = pd.read_csv(os.path.join(path_txt, name), delimiter="\t", header=0,
                              names=["time", "name", "emotion", "vad"])
            aux_data = pd.concat([aux_data, aux])

            # get filenames with emotion auxilary data
            emo_data = aux_data[aux_data['name'].str.match('Ses')]
            emotions = np.unique(emo_data['emotion'])
            emo_root = setup_folder_structure(emotions, cwd)
            emo_dict = get_dict(emo_data)
            for wav in emo_dict.keys():
                emotion = emo_dict[wav]
                wav_path = get_wav_path(wav, session, cwd)
                move_wav(wav_path, emo_root, wav, emotion)
        print(session, "moved.\nWriting to emotion_data.csv")
        emotion_data = pd.concat([emotion_data, emo_data[["name", "emotion"]]])
        emotion_data.to_csv("./emotion_data.csv")

        print(session, "moved & wrote.")
