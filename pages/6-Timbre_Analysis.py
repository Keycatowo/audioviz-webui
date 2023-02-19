#%%
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd
from src.st_helper import convert_df, show_readme
from src.timbre_analysis import (
    spectral_centroid_analysis,
    rolloff_frequency_analysis,
    spectral_bandwidth_analysis,
    harmonic_percussive_source_separation
)


#%% 頁面說明
show_readme("docs/6-Timbre Analysis.md")

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
    st.subheader("Select a segment of the audio")
    # 建立一個滑桿，可以選擇聲音片段，使用時間長度為單位
    start_time, end_time = st.slider("Select a segment of the audio", 
        0.0, duration, 
        (0.0, duration), 
        0.01
    )
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

    # spectral_centroid_analysis
    st.subheader("Spectral Centroid Analysis")
    fig6_1, ax6_1, centroid_value = spectral_centroid_analysis(y_sub, sr)
    st.pyplot(fig6_1)
    st.write(centroid_value)
    st.download_button(
        label="Download spectral centroid data",
        data=convert_df(pd.DataFrame(centroid_value)),
        file_name="centroid.csv",
        mime="text/csv",
    )

    # rolloff_frequency_analysis
    st.subheader("Rolloff Frequency Analysis")
    roll_percent = st.selectbox("Select rolloff frequency", [0.90, 0.95, 0.99])
    fig6_2, ax6_2, rolloff_value = rolloff_frequency_analysis(y_sub, sr, roll_percent=roll_percent)
    st.pyplot(fig6_2)
    st.write(rolloff_value)
    st.download_button(
        label="Download rolloff frequency data",
        data=convert_df(pd.DataFrame(rolloff_value)),
        file_name="rolloff.csv",
        mime="text/csv",
    )

    # spectral_bandwidth_analysis
    st.subheader("Spectral Bandwidth Analysis")
    fig6_3, ax6_3, bandwidth_value = spectral_bandwidth_analysis(y_sub, sr)
    st.pyplot(fig6_3)
    st.write(bandwidth_value)
    st.download_button(
        label="Download spectral bandwidth data",
        data=convert_df(pd.DataFrame(bandwidth_value)),
        file_name="bandwidth.csv",
        mime="text/csv",
    )

    # harmonic_percussive_source_separation
    st.subheader("Harmonic Percussive Source Separation")
    fig6_4, ax6_4 = harmonic_percussive_source_separation(y_sub, sr)
    st.pyplot(fig6_4)
    