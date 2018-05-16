import os
import sys
import shutil
import numpy as np


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

        if not os.path.exists("./data/iemocap/wav_val/" + emo_used[i]):
            os.makedirs("./data/iemocap/wav_val/" + emo_used[i])
        else: pass
        for n in p[val_onset:val_offset]:
            val.append(wav_list[n])
            src = "./data/iemocap/emotions/" + emo_used[i] + "/" + wav_list[n]
            dst = "./data/iemocap/wav_val/" + emo_used[i]
            shutil.copy(src, dst)

        if not os.path.exists("./data/iemocap/wav_test/" + emo_used[i]):
            os.makedirs("./data/iemocap/wav_test/" + emo_used[i])
        else: pass
        for n in p[test_onset:test_offset]:
            test.append(wav_list[n])
            src = "./data/iemocap/emotions/" + emo_used[i] + "/" + wav_list[n]
            dst = "./data/iemocap/wav_test/" + emo_used[i]
            shutil.copy(src, dst)


    print("number of training files:",len(train))
    print("number of validation files:",len(val))
    print("number of testing files:",len(test))
split(train_perc = 0.85, val_perc = 0.10, test_perc = 0.05)
