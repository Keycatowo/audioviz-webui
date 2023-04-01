#%%
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd
from src.st_helper import convert_df, show_readme, get_shift
from src.timbre_analysis import (
    spectral_centroid_analysis,
    rolloff_frequency_analysis,
    spectral_bandwidth_analysis,
    harmonic_percussive_source_separation
)

st.title("Timbre Analysis")
#%% 除錯訊息
if st.session_state.debug:
    st.write(st.session_state)

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
        y, sr = librosa.load(file, sr=22050)
        st.write(f"Sample rate: `{sr}`")
        duration = float(np.round(len(y)/sr-0.005, 2)) # 時間長度，取小數點後2位，向下取整避免超過音檔長度
        st.write(f"Duration(s): `{duration}`")
        y_all = y
        start_time = 0
        end_time = duration
        
#%% 片段模式
if file is not None:
    use_segment = st.sidebar.checkbox("使用片段模式", value=st.session_state["use_segment"], key="segment")
    st.session_state["use_segment"] = use_segment
    
    if use_segment:
        # 若使用片段模式，則顯示選擇片段的起始時間與結束時間
        if st.session_state.first_run:
            start_time = st.sidebar.number_input("開始時間", value=0.0, min_value=0.0, max_value=duration, step=0.01)
            end_time = st.sidebar.number_input("結束時間", value=duration, min_value=0.0, max_value=duration, step=0.01)
            st.session_state.first_run = False
            st.session_state.start_time = start_time
            st.session_state.end_time = end_time
        else:
            start_time = st.sidebar.number_input("開始時間", value=st.session_state.start_time, min_value=0.0, max_value=duration, step=0.01)
            end_time = st.sidebar.number_input("結束時間", value=st.session_state.end_time, min_value=0.0, max_value=duration, step=0.01)
            st.session_state.start_time = start_time
            st.session_state.end_time = end_time

    else:
        start_time = 0.0
        end_index = duration
    # 根據選擇的聲音片段，取出聲音資料
    start_index = int(start_time*sr)
    end_index = int(end_time*sr)
    if end_index <= start_index:
        st.sidebar.warning("結束時間必須大於開始時間", icon="⚠️")

    y_sub = y_all[start_index:end_index]
    x_sub = np.arange(len(y_sub))/sr
    
    if use_segment: 
        with st.expander("聲音片段(Segment of the audio)"):
            st.write(f"Selected segment: `{start_time}` ~ `{end_time}`, duration: `{end_time-start_time}`")
            st.audio(y_sub, format="audio/ogg", sample_rate=sr)

#%%
if file is not None:

    tab1, tab2, tab3, tab4 = st.tabs(["Spectral Centroid", "Rolloff Frequency", "Spectral Bandwidth", "Harmonic Percussive Source Separation"])

    shift_time, shift_array = get_shift(start_time, end_time) # shift_array為y_sub的時間刻度

    # spectral_centroid_analysis
    with tab1:
        st.subheader("Spectral Centroid Analysis")
        fig6_1, ax6_1, centroid_value = spectral_centroid_analysis(y_sub, sr, shift_array)
        st.pyplot(fig6_1)
        
        df_centroid = pd.DataFrame(centroid_value.T, columns=["Time(s)", "Centroid"])
        df_centroid["Time(s)"] = df_centroid["Time(s)"] + shift_time
        st.dataframe(df_centroid, use_container_width=True)
        st.download_button(
            label="Download spectral centroid data",
            data=convert_df(df_centroid),
            file_name="centroid.csv",
            mime="text/csv",
        )

    # rolloff_frequency_analysis
    with tab2:
        st.subheader("Rolloff Frequency Analysis")
        roll_percent = st.selectbox("Select rolloff frequency", [0.90, 0.95, 0.99])
        fig6_2, ax6_2, rolloff_value = rolloff_frequency_analysis(y_sub, sr, roll_percent=roll_percent, shift_array=shift_array)
        st.pyplot(fig6_2)
        df_rolloff = pd.DataFrame(rolloff_value.T, columns=["Time(s)", "Rolloff", "Rolloff_min"])
        df_rolloff["Time(s)"] = df_rolloff["Time(s)"] + shift_time
        st.dataframe(df_rolloff, use_container_width=True)
        st.download_button(
            label="Download rolloff frequency data",
            data=convert_df(df_rolloff),
            file_name="rolloff.csv",
            mime="text/csv",
        )

    # spectral_bandwidth_analysis
    with tab3:
        st.subheader("Spectral Bandwidth Analysis")
        fig6_3, ax6_3, bandwidth_value = spectral_bandwidth_analysis(y_sub, sr, shift_array)
        st.pyplot(fig6_3)
        df_bandwidth = pd.DataFrame(bandwidth_value.T, columns=["Time(s)", "Bandwidth"])
        df_bandwidth["Time(s)"] = df_bandwidth["Time(s)"] + shift_time
        st.dataframe(df_bandwidth, use_container_width=True)
        st.download_button(
            label="Download spectral bandwidth data",
            data=convert_df(df_bandwidth),
            file_name="bandwidth.csv",
            mime="text/csv",
        )

    # harmonic_percussive_source_separation
    with tab4:
        st.subheader("Harmonic Percussive Source Separation")
        fig6_4, ax6_4, (Harmonic_data) = harmonic_percussive_source_separation(y_sub, sr, shift_array)
        D, H, P, t = Harmonic_data
        st.pyplot(fig6_4)

        st.download_button(
            label="Download Full power spectrogram data",
            data=convert_df(pd.DataFrame(D)),
            file_name="Full_power_spectrogram.csv",
            use_container_width=True,
        )
        st.download_button(
            label="Download Harmonic power spectrogram data",
            data=convert_df(pd.DataFrame(H)),
            file_name="Harmonic_power_spectrogram.csv",
            use_container_width=True,
        )
        st.download_button(
            label="Download Percussive power spectrogram data",
            data=convert_df(pd.DataFrame(P)),
            file_name="Percussive_power_spectrogram.csv",
            use_container_width=True,
        )
        st.download_button(
            label="Download Time data",
            data=convert_df(pd.DataFrame(t+shift_time, columns=["Time(s)"])),
            file_name="Time_scale.csv",
            use_container_width=True,
        )