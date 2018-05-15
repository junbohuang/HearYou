import os
import shutil
import sys
import pandas as pd
import numpy as np

from infrastructure.helper_function import *

# loop through folders
cwd = os.getcwd()
sessions = [session for session in os.listdir("./data/iemocap/IEMOCAP_full_release/") if session.startswith('Session')]
for session in sessions:
    path_txt = "./data/iemocap/IEMOCAP_full_release/" + session + "/dialog/EmoEvaluation"
    path_wav = "./data/iemocap/IEMOCAP_full_release/" + session + "/sentences/wav"
    filenames_txt = [txt for txt in os.listdir(path_txt) if txt.startswith('Ses') and txt.endswith('.txt')]
    filenames_wav = [wav for wav in os.listdir(path_wav) if wav.startswith('Ses') and wav.endswith('.wav')]
    aux_data = pd.DataFrame(columns=["time", "name", "emotion","vad"])

    for name in filenames_txt:
        aux = pd.read_csv(os.path.join(path_txt, name),delimiter="\t",header=0, names=["time", "name", "emotion","vad"], engine='c')
        aux_data = pd.concat([aux_data,aux])
        
        # get filenames with emotion auxilary data
        emo_data = aux_data[aux_data['name'].str.match('Ses')]
        # print(emo_data)
        emotions = np.unique(emo_data['emotion'])
        # print(emotions)
        emo_root = setup_folder_structure(emotions, cwd)
        # print(emo_root)
        emo_dict = get_dict(emo_data)
        # print(emo_dict)
        for wav in emo_dict.keys(): 
            emotion = emo_dict[wav]
            wav_path = get_wav_path(wav, session, cwd)
            move_wav(wav_path, emo_root, wav, emotion)
        print(session, name,"moved.")
    print(session, "moved")
print("folder structure all set up!")
   
