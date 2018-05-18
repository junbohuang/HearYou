import os
import shutil
import sys
import pandas as pd
import numpy as np

from infrastructure.helper_function import *


def main():
    if not os.path.exists("./data/iemocap/wav_train/"):
        split(train_perc = 0.85, val_perc = 0.10)
    else: pass

    # loop through folders
    cwd = os.getcwd()
    sessions = [session for session in os.listdir(cwd + "/data/iemocap/IEMOCAP_full_release/") 
                     if session.startswith('Session')]
    
    emotion_data= pd.DataFrame(columns=["name", "emotion"])

    for session in sessions:
        path_txt = cwd + "/data/iemocap/IEMOCAP_full_release/" + session + "/dialog/EmoEvaluation"
        path_wav = cwd + "/data/iemocap/IEMOCAP_full_release/" + session + "/sentences/wav"
        filenames_txt = [txt for txt in os.listdir(path_txt) if txt.startswith('Ses') and txt.endswith('.txt')]
        filenames_wav = [wav for wav in os.listdir(path_wav) if wav.startswith('Ses') and wav.endswith('.wav')]
        aux_data = pd.DataFrame(columns=["time", "name", "emotion","vad"])

        for name in filenames_txt:
            aux = pd.read_csv(os.path.join(path_txt, name),delimiter="\t",header=0, 
                             names=["time", "name", "emotion","vad"])
            aux_data = pd.concat([aux_data,aux])

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
        emotion_data = pd.concat([emotion_data, emo_data[["name","emotion"]]])
        emotion_data.to_csv("./emotion_data.csv") 

        print(session, "moved & wrote.")

    
if __name__ == "__main__":
    main()
    print("folder structure all set up!")

    print("plotting ...")
    emo_dir = "./data/iemocap/emotions"
    emotions = [emotion for emotion in os.listdir(emo_dir) if not emotion.startswith('.')]
    emo_num = {}
    for emotion in emotions:
        emo_num[emotion] = len(os.listdir(emo_dir + "/" + emotion))
    plot_emo_num(emo_num, server="local")
    print("emotion category distribution plot saved!")

        
        

         
