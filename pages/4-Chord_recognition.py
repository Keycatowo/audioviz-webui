#%%
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd
from src.st_helper import convert_df, show_readme
from src.chord_recognition import (
    plot_chord_recognition,
    plot_binary_template_chord_recognition,
    chord_table
)

st.title("Chord Recognition")

#%% 頁面說明
show_readme("docs/1-Basic Information.md")


#%% 上傳檔案區塊
with st.expander("上傳檔案(Upload Files)"):
    file = st.file_uploader("Upload your music library", type=["mp3", "wav", "ogg"])

    if file is not None:
        st.audio(file, format="audio/ogg")
        st.subheader("File information")
        st.write(f"File name: `{file.name}`", )
        st.write(f"File type: `{file.type}`")
        st.write(f"File size: `{file.size}`")

        # 載入音檔
        y, sr = librosa.load(file, sr=44100)
        st.write(f"Sample rate: `{sr}`")
        duration = float(np.round(len(y)/sr-0.005, 2)) # 時間長度，取小數點後2位，向下取整避免超過音檔長度
        st.write(f"Duration(s): `{duration}`")

        y_all = y

#%%
if file is not None:

    ### Start of 選擇聲音片段 ###
    with st.expander("選擇聲音片段(Select a segment of the audio)"):
        
        # 建立一個滑桿，可以選擇聲音片段，使用時間長度為單位
        start_time, end_time = st.slider("Select a segment of the audio", 
            0.0, duration, 
            (st.session_state.start_time, duration), 
            0.01
        )
        st.session_state.start_time = start_time

    st.write(f"Selected segment: `{start_time}` ~ `{end_time}`, duration: `{end_time-start_time}`")

    # 根據選擇的聲音片段，取出聲音資料
    start_index = int(start_time*sr)
    end_index = int(end_time*sr)
    y_sub = y_all[start_index:end_index]

        
    # 建立一個y_sub的播放器
    st.audio(y_sub, format="audio/ogg", sample_rate=sr)
    # 計算y_sub所對應時間的x軸
    x_sub = np.arange(len(y_sub))/sr
    ### End of 選擇聲音片段 ###

    tab1, tab2 = st.tabs(["Chord Recognition", "Chord Recognition with Binary Template"])
    
    # plot_chord_recognition 
    with tab1:
        st.subheader("plot_chord_recognition")
        fig4_1, ax4_1, chord_max = plot_chord_recognition(y_sub, sr)
        st.pyplot(fig4_1)
        chord_results = chord_table(chord_max)
        chord_results_df = pd.DataFrame(chord_results)
        st.write(chord_results_df)
        

    # plot_binary_template_chord_recognition
    with tab2:
        st.subheader("plot_binary_template_chord_recognition")
        fig4_2, ax4_2 = plot_binary_template_chord_recognition(y_sub, sr)
        st.pyplot(fig4_2)


