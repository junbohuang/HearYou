import os
import shutil
import sys
import pandas as pd
import numpy as np

from helper_function import *



def main():

    if not os.path.exists("./emotion_data.csv"):
        get_csv()
    else: pass

    if not os.path.exists("./data/iemocap/wav_train/") and \
            os.path.exists("./data/iemocap/wav_valid/") and \
            os.path.exists("./data/iemocap/wav_test/"):
        split(train_perc = 0.85, val_perc = 0.10)
    else: pass


    
if __name__ == "__main__":
    main()
    print("folder structure all set up!")

    print("plotting ...")
    emo_dir = "./data/iemocap/emotions"
    emotions = [emotion for emotion in os.listdir(emo_dir) if not emotion.startswith('.')]
    emo_num = {}
    for emotion in emotions:
        emo_num[emotion] = len(os.listdir(emo_dir + "/" + emotion))
    plot_emo_num(emo_num)
    print("emotion category distribution plot saved!")

        
        

         
