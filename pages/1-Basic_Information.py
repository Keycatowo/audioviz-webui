#%%
import streamlit as st
from src.basic_info import plot_waveform, signal_RMS_analysis
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd


@st.experimental_memo
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

#%%
with st.expander("頁面說明(Page Description)"):
    with open("docs/1-Basic Information.md", "r", encoding="utf-8") as f:
        st.markdown(f.read())

#%%
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
        
        # st.write(f"{type(file)}")
        # st.write(f"{type(y)}")
        # st.write(f"{type(sr)}")

        y_all = y

#%%
if file is not None:

    # 選擇聲音片段
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


    # 繪製聲音波形圖
    st.subheader("Waveform(mathplotlib)")
    fig1_1, ax_1_1 = plt.subplots()
    ax_1_1.plot(x_sub, y_sub)
    ax_1_1.set_xlabel("Time(s)")
    ax_1_1.set_ylabel("Amplitude")
    ax_1_1.set_title("Waveform")
    st.pyplot(fig1_1)
    
    # 繪製聲音波形圖
    st.subheader("Waveform(plotly)")
    fig1_2 = go.Figure(data=go.Scatter(x=x_sub, y=y_sub))
    fig1_2.update_layout(
        title="Waveform",
        xaxis_title="Time(s)",
        yaxis_title="Amplitude",
    )
    st.plotly_chart(fig1_2)

    # 繪製聲音RMS圖
    st.subheader("signal_RMS_analysis")
    fig1_3, ax1_3, times, rms = signal_RMS_analysis(y_sub)
    st.pyplot(fig1_3)
    # st.write("Times of RMS:", times)
    # st.write("RMS:", rms)

    # 繪製聲音Spectrogram圖(使用librosa繪製)
    st.subheader("Spectrogram")
    stft = librosa.stft(y_sub)
    stft_db = librosa.amplitude_to_db(abs(stft))
    # add a figure
    fig1_4, ax1_4 = plt.subplots()
    ax1_4 = librosa.display.specshow(stft_db, x_axis='time', y_axis='log')
    st.pyplot(fig1_4)

    # 提供一個下載按鈕，讓使用者可以下載RMS資料
    st.subheader("Download RMS data")
    
    col1, col2 = st.columns(2)
    with col1:
        rms_df = pd.DataFrame({"Time": times, "RMS": rms[0,:]})
        st.write(rms_df)
    with col2:
        st.download_button(
            "Doanload RMS data",
            convert_df(rms_df),
            "rms.csv",
            "text/csv",
            key="download-csv"
        )
        