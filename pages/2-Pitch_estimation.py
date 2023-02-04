#%%
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd
from src.pitch_estimation import plot_mel_spectrogram, plot_constant_q_transform, pitch_class_type_one_vis
from src.st_helper import convert_df, show_readme

#%% 頁面說明
#%% 頁面說明
show_readme("docs/2-Pitch_estimation.md")


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

#%% 功能區塊
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

    # Mel-frequency spectrogram
    st.subheader("Mel-frequency spectrogram")
    with_pitch = st.checkbox("Show pitch", value=True)
    fig2_1, ax2_1 = plot_mel_spectrogram(y_sub, sr, with_pitch=with_pitch)
    st.pyplot(fig2_1)

    # Constant-Q transform
    st.subheader("Constant-Q transform")
    fig2_2, ax2_2 = plot_constant_q_transform(y_sub, sr)
    st.pyplot(fig2_2)
    
    # chroma
    st.subheader("Chroma")
    chroma = librosa.feature.chroma_stft(y=y_sub, sr=sr)
    st.write(chroma)

    # Pitch class type one
    st.subheader("Pitch class type one")
    fig2_3, ax2_3, df_pitch_class = pitch_class_type_one_vis(y_sub, sr)
    st.pyplot(fig2_3)
    st.write(df_pitch_class)
    st.download_button(
        label="Download pitch class type one",
        data=convert_df(pd.DataFrame(df_pitch_class)),
        file_name="pitch_class_type_one.csv",
        mime="text/csv",
    )