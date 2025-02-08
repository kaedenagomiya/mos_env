import toybox
import streamlit as st
from streamlit import config

#st.write(f'{config.get_option("browser.gatherUsageStats")=}')

#st.title("Mean Opinion Score(MOS) evaluation")
#st.write('same message')


#st.title('answer form')

#st.audio("./tmp_wav/text_mako.wav", format="audio/wav")
audio_bytes = toybox.load_audio("./tmp_wav/text_mako.wav")
st.audio(audio_bytes, format="audio/wav") # you can set sample_rate option


#with st.form(key='survey_form'):
#    #st.audio("./tmp_wav/text_mako.wav", format="audio/wav")
#    audio_bytes = toybox.load_audio("./tmp_wav/text_mako.wav")
#    st.audio(audio_bytes, format="audio/wav") # you can set sample_rate option

#    opinion = st.radio(
#        'Please rate the speech.', 
#        ['1', '2', '3', '4', '5']
#    )
    
#    submit_button = st.form_submit_button("submit")