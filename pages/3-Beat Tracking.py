#%%
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd
from src.beat_track import onsets_detection, plot_onset_strength, beat_analysis, predominant_local_pulse, static_tempo_estimation, plot_tempogram
from src.st_helper import convert_df, show_readme
import numpy as np

st.title('Beat Tracking')

#%% 頁面說明
show_readme("docs/3-Beat Tracking.md")

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

    # onsets_detection
    st.subheader("onsets_detection")
    fig3_1, ax3_1, y_onset_clicks = onsets_detection(y_sub, sr)
    st.pyplot(fig3_1)
    st.audio(y_onset_clicks, format="audio/ogg", sample_rate=sr)

    # onset_strength
    st.subheader("onset_strength")
    onset_strength_standard = st.checkbox("standard", value=True)
    onset_strength_custom_mel = st.checkbox("custom_mel", value=False)
    onset_strength_cqt = st.checkbox("cqt", value=False)
    fig3_2, ax3_2 = plot_onset_strength(y_sub, sr,
        standard=onset_strength_standard,
        custom_mel=onset_strength_custom_mel,
        cqt=onset_strength_cqt
    )
    st.pyplot(fig3_2)

    # beat_analysis
    st.subheader("beat_analysis")
    spec_type = st.selectbox("spec_type", ["mel", "stft"])
    spec_hop_length = st.number_input("spec_hop_length", value=512)
    fig3_3, ax3_3, y_beats = beat_analysis(y_sub, sr,
        spec_type=spec_type,
        spec_hop_length=spec_hop_length
    )
    st.pyplot(fig3_3)


    # predominant_local_pulse
    st.subheader("predominant_local_pulse")
    fig3_4, ax3_4 = predominant_local_pulse(y_sub, sr)
    st.pyplot(fig3_4)

    # static_tempo_estimation
    st.subheader("static_tempo_estimation")
    static_tempo_estimation_hop_length = st.number_input("hop_length", value=512)
    fig3_5, ax3_5 = static_tempo_estimation(y_sub, sr,
        hop_length=static_tempo_estimation_hop_length
    )
    st.pyplot(fig3_5)

    # Tempogram
    st.subheader("Tempogram")
    tempogram_type = st.selectbox("tempogram_type", ["fourier", "autocorr"], index=1)
    tempogram_hop_length = st.number_input("Tempogram_hop_length", value=512)
    fig3_6, ax3_6 = plot_tempogram(y_sub, sr,
        type=tempogram_type,
        hop_length=tempogram_hop_length
    )
    st.pyplot(fig3_6)