"""
uv run streamlit run test_w_ljwav_v1.py --server.port 8051
"""
# play audio
# eval audio quality
# save result
import json
import datetime
import uuid
import random
from pathlib import Path
import pandas as pd
import streamlit as st

import toybox

# test_with_LJ

# test with LJ
#max_num4play = 10
#config_wav_path = "configs/test_dataset.json"
#test_config_list = toybox.load_json(config_wav_path)[0]
#{'name': 'LJ031-0171', 'wav_path': 'data/ljspeech/LJSpeech-1.1/wavs/LJ031-0171.wav', 'text': "", 'phonemes': []}

#audioinfo_list = [test_config_list[i]['name'] for i in range(len(test_config_list))]
#audioinfo_list = audioinfo_list[1:max_num4play]
#audiopath_list = [f"./data/ljspeech/LJSpeech-1.1/wavs/{wav_name}.wav" for wav_name in audioinfo_list]
#length_evalset = len(audiopath_list)

#exp_list = ['a', 'b', 'c']
#exp_index = random.sample(exp_list, 1)[0]

# load only once
if "exp_index" not in st.session_state:
    exp_list = ['a', 'b', 'c']
    st.session_state.exp_index = random.choice(exp_list)

exp_index = st.session_state.exp_index

if "audiopath_list" not in st.session_state:
    exp_config_path = f"./exp_configs/config_mos_{exp_index}.csv"
    df = pd.read_csv(exp_config_path)
    st.session_state.audiopath_list = df['wav_path'].tolist()

audiopath_list = st.session_state.audiopath_list
length_evalset = len(audiopath_list)
#print(audiopath_list['wav_path'][0])

# wav directory
# LJ_DIR = "./data/ljspeech/LJSpeech-1.1/wavs"


# init application state
# for exp

# current page number
if "current_page" not in st.session_state:
    st.session_state.current_page = 0
# current audio index
if "current_audio_index" not in st.session_state:
    st.session_state.current_audio_index = 0
# current evaluations result
if "audio_evaluations" not in st.session_state:
    st.session_state.audio_evaluations = []


def page_audio_evaluation():
    st.header(f"Mean Opinion Score for evaluating speech EX:{exp_index}")

    # get current audio index
    current_audio_index = st.session_state.current_audio_index

    if current_audio_index < length_evalset:
        st.write(f"Play the following audio_{current_audio_index+1}/{length_evalset} and rate the quality on a 5-point scale.")
        st.write(f"wav_path: {audiopath_list[current_audio_index]}")
        # load and play audio
        audio_bytes = toybox.load_audio2bin(audiopath_list[current_audio_index])
        st.audio(audio_bytes, format="audio/wav")
        #st.audio(audiopath_list[current_audio_index], format="audio/wav")
        # evaluate audio
        evaluation = st.radio(
            f"Please choose from below ( **5=very good**, 1=very bad ) for audio_{current_audio_index+1} quality",
            [5, 4, 3, 2, 1],
            index=2,
            horizontal=False,
            key=f"evaluation_{current_audio_index}"
        )

        # button for next
        if st.button("Next"):
            # save currently
            st.session_state.audio_evaluations.append({
                "audio_path": audiopath_list[current_audio_index],
                "evaluation": evaluation,
                "timestamp": datetime.datetime.now()
            })
            # move page
            st.session_state.current_audio_index += 1
            st.rerun()
    else:
        st.success("fin all audio evaluation")
        if st.button("save and confirm results"):
            # save results
            st.write(st.session_state.audio_evaluations)
            #save_results_to_csv(st.session_state.audio_evaluations)
            st.write("Thank you for your evaluation. Save All results.")


# 

if __name__ == "__main__":
    page_audio_evaluation()