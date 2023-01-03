#%%
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from src.basic_info import plot_waveform, signal_RMS_analysis

#%%
st.header("Music Analysis Tool")
st.subheader("This tool will help you analyze your music library")

#%%
with st.expander("上傳檔案(Upload Files)"):
    file = st.file_uploader("Upload your music library", type=["mp3", "wav", "ogg"])

if file is not None:
    st.subheader("Audio")
    audio_file = st.audio(file, format="audio/ogg")
    st.subheader("File information")
    st.write(f"File name: `{file.name}`", )
    st.write(f"File type: `{file.type}`")
    st.write(f"File size: `{file.size}`")
    
    # 載入音檔
    y, sr = librosa.load(file, sr=44100)
    st.write(f"Sample rate: `{sr}`")
    st.write(f"Duration: `{len(y)/sr}`")

    # 繪製聲音波形圖
    st.subheader("Waveform")
    fig, ax = plt.subplots()
    ax.plot(y)

    st.pyplot(fig)
        
    # 繪製聲音頻譜圖
    st.subheader("signal_RMS_analysis")
    fig, ax = signal_RMS_analysis(y)
    st.pyplot(fig)