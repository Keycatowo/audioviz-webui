#%%
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd
from src.st_helper import convert_df, show_readme, get_shift
from src.basic_info import plot_waveform, signal_RMS_analysis

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

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Waveform(mathplotlib)",
        "Waveform(plotly)",
        "signal_RMS_analysis",
        "Spectrogram",
        "Download RMS data"])
    
    shift_time, shift_array = get_shift(start_time, end_time) # shift_array為y_sub的時間刻度

    # 繪製聲音波形圖
    with tab1:
        st.subheader("Waveform(mathplotlib)")
        fig1_1, ax_1_1 = plt.subplots()
        ax_1_1.plot(x_sub + shift_time, y_sub)
        ax_1_1.set_xlabel("Time(s)")
        ax_1_1.set_ylabel("Amplitude")
        ax_1_1.set_title("Waveform")
        st.pyplot(fig1_1)
    
    # 繪製聲音波形圖
    with tab2:
        st.subheader("Waveform(plotly)")
        fig1_2 = go.Figure(data=go.Scatter(x=x_sub + shift_time, y=y_sub))
        fig1_2.update_layout(
            title="Waveform",
            xaxis_title="Time(s)",
            yaxis_title="Amplitude",
        )
        st.plotly_chart(fig1_2)

    # 繪製聲音RMS圖
    with tab3:
        st.subheader("signal_RMS_analysis")
        fig1_3, ax1_3, times, rms = signal_RMS_analysis(y_sub, shift_time=shift_time)
        st.pyplot(fig1_3)   

    # 繪製聲音Spectrogram圖(使用librosa繪製)
    with tab4:
        st.subheader("Spectrogram")
        stft = librosa.stft(y_sub)
        stft_db = librosa.amplitude_to_db(abs(stft))
        # add a figure
        fig1_4, ax1_4 = plt.subplots()
        librosa.display.specshow(stft_db, x_axis='time', y_axis='log', sr=sr, ax=ax1_4)
        ax1_4.set_xticks(shift_array - shift_array[0],
                         shift_array)
        ax1_4.autoscale()
        st.pyplot(fig1_4)

    # 下載RMS資料
    with tab5:
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
        
# %%
