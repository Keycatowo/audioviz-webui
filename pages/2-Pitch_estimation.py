#%%
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd
import seaborn as sns
from src.st_helper import convert_df, show_readme, get_shift
from src.pitch_estimation import plot_mel_spectrogram, plot_constant_q_transform, pitch_class_type_one_vis, pitch_class_histogram_chroma

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

    tab1, tab2, tab3, tab4 = st.tabs(["Mel-frequency spectrogram", "Constant-Q transform", "Chroma", "Pitch class"])

    shift_time, shift_array = get_shift(start_time, end_time) # shift_array為y_sub的時間刻度

    # Mel-frequency spectrogram
    with tab1:
        st.subheader("Mel-frequency spectrogram")
        with_pitch = st.checkbox("Show pitch", value=True)
        fig2_1, ax2_1 = plot_mel_spectrogram(y_sub, sr, shift_array, with_pitch)
        st.pyplot(fig2_1)

    # Constant-Q transform
    with tab2:
        st.subheader("Constant-Q transform")
        fig2_2, ax2_2 = plot_constant_q_transform(y_sub, sr, shift_array)
        st.pyplot(fig2_2)
    
    # chroma
    with tab3:
        st.subheader("Chroma")
        
        chroma = librosa.feature.chroma_stft(y=y_sub, sr=sr)
        chroma_t = librosa.times_like(chroma, sr)
        df_chroma = pd.DataFrame(chroma)
        df_chroma_t = pd.DataFrame({"Time(s)": chroma_t})
        df_chroma_t["Time(frame)"] = list(range(len(chroma_t)))
        df_chroma_t["Time(s)"] = df_chroma_t["Time(s)"] + shift_time
        df_chroma_t = df_chroma_t[["Time(frame)", "Time(s)"]]
        
        fig2_3, ax2_3 = plt.subplots(figsize=(10, 4))
        sns.heatmap(chroma, ax=ax2_3)
        ax2_3.set_title("Chroma")
        ax2_3.set_xlabel("Time(frame)")
        ax2_3.invert_yaxis()
        st.pyplot(fig2_3)
        
        st.write("Chroma value")
        st.dataframe(df_chroma, use_container_width=True)
        st.download_button(
            label="Download chroma",
            data=convert_df(df_chroma),
            file_name="chroma_value.csv",
        )
        st.write("Chroma time")
        st.dataframe(df_chroma_t, use_container_width=True)
        st.download_button(
            label="Download chroma time",
            data=convert_df(df_chroma_t),
            file_name="chroma_time.csv",
        )

    # Pitch class type one
    with tab4:
        st.subheader("Pitch class(chroma)")
        high_res = st.checkbox("High resolution", value=False)
        fig2_4, ax2_4, df_pitch_class = pitch_class_histogram_chroma(y_sub, sr, high_res)
        st.pyplot(fig2_4)
        st.write(df_pitch_class)
        st.download_button(
            label="Download pitch class(chroma)",
            data=convert_df(pd.DataFrame(df_pitch_class)),
            file_name="Pitch_class(chroma).csv",
            mime="text/csv",
        )