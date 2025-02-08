import os
import json
import datetime
import random
import uuid
from pathlib import Path
import pandas as pd
import streamlit as st

import toybox

#max_num4play = 2
#config_wav_path = "configs/test_dataset.json"
#test_config_list = toybox.load_json(config_wav_path)[0]
#{'name': 'LJ031-0171', 'wav_path': 'data/ljspeech/LJSpeech-1.1/wavs/LJ031-0171.wav', 'text': "", 'phonemes': []}
#test_audioname_list = [test_config_list[i]['name'] for i in range(len(test_config_list))]
#test_audioname_list = test_audioname_list[1:max_num4play]
#test_wavpath_list=

SAVE_DIR = './results_evaluation'

# wav directory
# LJ_DIR = "./data/ljspeech/LJSpeech-1.1/wavs"

# load only once ---------------------------------------------
if "model_names_list" not in st.session_state:
    #st.session_state.model_names_list = ['ljspeech', 'gradtts', 'gradseptts', 'gradtfktts', 'gradtfk5tts', 'nix_deter']
    st.session_state.model_names_list = ['ljspeech', 'gradtts', 'gradtfktts', 'gradtfk5tts', 'nix_deter']

model_names_list = st.session_state.model_names_list

if "exp_version" not in st.session_state:
    exp_list = ['a', 'b', 'c']
    st.session_state.exp_version = random.choice(exp_list)

exp_version = st.session_state.exp_version

if "audiopath_list" not in st.session_state:
    exp_config_path = f"./exp_configs/config_mos_{exp_version}.csv"
    df = pd.read_csv(exp_config_path)
    st.session_state.audiopath_list = df['wav_path'].tolist()

audiopath_list = st.session_state.audiopath_list
#print(audiopath_list)
length_evalset = len(audiopath_list)

if "explanation_list" not in st.session_state:
    st.session_state.explanation_list = toybox.load_yaml(f"./exp_configs/explanation4exp.yaml")

explanation_list = st.session_state.explanation_list
#print(explanation_list)


# init application state ---------------------------------------
# current page number
if "current_page" not in st.session_state:
    st.session_state.current_page = 0
# current audio index
if "current_audio_index" not in st.session_state:
    st.session_state.current_audio_index = 0
# current evaluations result
if "audio_evaluations" not in st.session_state:
    st.session_state.audio_evaluations = []
# user info
if "user_info" not in st.session_state:
    st.session_state.user_info = {}
# unique user id
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

user_id = st.session_state.user_id

if "confirmed_evaluation" not in st.session_state:
    st.session_state.confirmed_evaluation = False

# page_user_registoration ---------------------------------------
def page_user_registoration():
    st.header("Subject Registration Form")
    st.write("## **Please enter the following information**")

    st.markdown(f"{explanation_list['text4agreement']}")

    # check agreement
    user_name = st.text_input("**your name**", key="user_name")
    agree = st.checkbox("I have reviewed the experiment and agree", key="user_consent")

    if not agree:
        st.warning("You need to agree to the experiment")

    if st.button("Next"):
        if not user_name:
            st.error("you must enter your name if you agree to the experiment.")
        elif not agree:
            st.error("You must agree to the content of the experiment.")
        else:
            # save subject information
            # create dir
            user_dir = toybox.create_user_dir(user_id, SAVE_DIR)
            user_info_file = os.path.join(user_dir, "user_info.csv")

            st.session_state.user_info = {
                "user_id": st.session_state.user_id,
                "name": user_name,
                "consent": agree,
                "exp_version": exp_version,
                "registration_timestamp": datetime.datetime.now()
            }
            ###
            toybox.save_to_csv([st.session_state.user_info], user_info_file)
            #st.success(
            st.code(f"Subject information saved:\nyour user_id is {st.session_state.user_id}.")
            # go to the next page
            st.session_state.current_page += 1
            # raise flag for update page
            #st.session_state.need_rerun = True
            st.rerun()


# page_describe_experiment ---------------------------------------
def page_describe_experiment():
    st.header("How to do experiment for speech evaluation")

    st.markdown(f"{explanation_list['text4mosexperiment']}")

    st.warning(f"**Check above for experimental methods for speech evaluation.**  \n\
     Once confirmed, the MOS experiment will begin on the next page.")

    if st.button("Next"):
        # go to the next page
        st.session_state.current_page += 1
        # raise flag for update page
        #st.session_state.need_rerun = True
        st.rerun()


# page_audio_evaluation ---------------------------------------
def page_audio_evaluation():
    ###
    #st.header(f"Mean Opinion Score for evaluating speech EX:{exp_version}")
    st.header(f"Mean Opinion Score for evaluating speech")
    
    # get current audio index
    current_audio_index = st.session_state.current_audio_index

    if current_audio_index < length_evalset:
        # page for evaluation audio
        ###
        #st.write(f"wav_path: {audiopath_list[current_audio_index]}")
        st.write(f"{explanation_list['table_mos4eval']}")
        st.write(f"## {current_audio_index+1}/{length_evalset}")
        st.write(f"Play the following audio and rate the quality on a 5-point scale.")
        ###
        #st.write(f"wav_path: {audiopath_list[current_audio_index]}")
        # load and play audio
        audio_bytes = toybox.load_audio2bin(audiopath_list[current_audio_index])
        st.audio(audio_bytes, format="audio/wav")
        # evaluate audio
        evaluation = st.radio(
            f"Please choose from below ( **5=Excellent**, **1=bad** ) for audio_{current_audio_index+1} quality",
            [5, 4, 3, 2, 1],
            index=2,
            horizontal=False,
            key=f"evaluation_{current_audio_index}"
        )

        st.checkbox(
            "Are you satisfied with your evaluation of this audio?",
            key="confirmed_evaluation",
        )
        #   value=st.session_state.confirmed_evaluation,

        confirmed_evaluation = st.session_state.confirmed_evaluation

        if not confirmed_evaluation:
            st.warning("Check this box to prevent unintended transitions caused by the Next button")
        #if confirmed_evaluation != st.session_state.confirmed_evaluation:
        #    st.session_state.confirmed_evaluation = confirmed_evaluation
        
        # button for next
        if st.button("Next"):
            if not confirmed_evaluation:
                st.error("Please confirm your evaluation before proceeding.")
            else:
                # save currently
                evaluation_result = {
                    "exp_version": exp_version,
                    "user_id": st.session_state.user_id,
                    "wav_path": audiopath_list[current_audio_index],
                    "model_name": toybox.extract_model_name(audiopath_list[current_audio_index], model_names_list),
                    "evaluation": evaluation,
                    "timestamp": datetime.datetime.now().isoformat() 
                }
                st.session_state.audio_evaluations.append(evaluation_result)
                # save audio evaluation to csv
                user_dir = toybox.create_user_dir(user_id, SAVE_DIR)
                audio_evaluations_path = os.path.join(user_dir, "audio_evaluations.csv")
                ###
                toybox.save_to_csv([evaluation_result], audio_evaluations_path)
                #move to next page
                st.session_state.current_audio_index += 1
                #st.session_state.confirmed_evaluation = False
                ### save current progress
                progress = {
                    #"current_audio_index": current_audio_index+1,
                    "current_audio_index": st.session_state.current_audio_index,
                    "audio_evaluations": st.session_state.audio_evaluations
                }
                toybox.save_progress(st.session_state.user_id, SAVE_DIR, progress)
                    #rerun the script to refresh the UI
                #st.session_state.need_rerun = True
                st.rerun()   
    else:
        st.success("fin all audio evaluation")
        st.write("Thank you for your evaluation. Save All results.")


# main_process ---------------------------------------
def main():
    # initialize subject dir
    user_dir = toybox.create_user_dir(st.session_state.user_id, SAVE_DIR)
    
    # load previous progress
    progress = toybox.load_progress(st.session_state.user_id, SAVE_DIR)
    if progress:
        st.session_state.current_audio_index = progress["current_audio_index"]
        st.session_state.audio_evaluations = progress["audio_evaluations"]

    # manage pagenation
    if st.session_state.current_page == 0:
        page_user_registoration()
    elif st.session_state.current_page == 1:
        page_describe_experiment()
    elif st.session_state.current_page == 2:
        page_audio_evaluation()

    #if ("need_rerun" in st.session_state) and st.session_state.need_rerun:
    #    st.session_state.need_rerun = False  # フラグをリセット
    #    st.rerun() 


# main ---------------------------------------
if __name__ == "__main__":
    main()
