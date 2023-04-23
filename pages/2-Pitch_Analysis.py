#%%
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd
import seaborn as sns
from src.st_helper import convert_df, get_shift, update_sessions, use_plotly
from src.pitch_estimation import (
    plot_mel_spectrogram, 
    plot_constant_q_transform, 
    plot_chroma,
    plot_pitch_class
)


st.title("Pitch Analysis")
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
        st.session_state["file_name"] = file.name
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

#%% 更新session
update_sessions()
        
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
            
    use_plotly()
            
            
#%% 功能分頁
if file is not None:

    tab1, tab2, tab3, tab4 = st.tabs(["Mel-frequency spectrogram", "Constant-Q transform", "Chroma", "Pitch class"])

    shift_time, shift_array = get_shift(start_time, end_time) # shift_array為y_sub的時間刻度

    # Mel-frequency spectrogram
    with tab1:
        st.subheader("Mel-frequency spectrogram")
        with_pitch = st.checkbox("Show pitch", value=st.session_state["2-Pitch"]["show_f0"])
        st.session_state["2-Pitch"]["show_f0"] = with_pitch
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
        if st.session_state["use_plotly"]:
            fig2_3, ax2_3, chroma, chroma_t = plot_chroma(y_sub, sr, shift_time, 12, True, use_plotly=True)
            st.plotly_chart(fig2_3)
        else:
            fig2_3, ax2_3, chroma, chroma_t = plot_chroma(y_sub, sr, shift_time, 12, True, use_plotly=False)
            st.pyplot(fig2_3)
        
        # 轉換成dataframe
        df_chroma = pd.DataFrame(chroma)
        df_chroma_t = pd.DataFrame({"Time(s)": chroma_t})
        df_chroma_t["Time(frame)"] = list(range(len(chroma_t)))
        df_chroma_t["Time(s)"] = df_chroma_t["Time(s)"] + shift_time
        df_chroma_t = df_chroma_t[["Time(frame)", "Time(s)"]]
        
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
        resolution_ratio = st.number_input("Use higher resolution", value=st.session_state["2-Pitch"]["resolution_ratio"], min_value=1, max_value=100, step=1)
        st.session_state["2-Pitch"]["resolution_ratio"] = resolution_ratio
        if st.session_state["use_plotly"]:
            fig2_4, ax2_4, df_pitch_class = plot_pitch_class(y_sub, sr, resolution_ratio=resolution_ratio, use_plotly=True, return_data=True)
            st.plotly_chart(fig2_4)
        else:
            fig2_4, ax2_4, df_pitch_class = plot_pitch_class(y_sub, sr, resolution_ratio=resolution_ratio, use_plotly=True, return_data=True)
            
        fig2_4, ax2_4, df_pitch_class = plot_pitch_class(
            y_sub, sr, 
            resolution_ratio=resolution_ratio,
            use_plotly = False,
            return_data = True,
        )
        st.pyplot(fig2_4)
        st.write(df_pitch_class)
        
        st.download_button(
            label="Download pitch class(chroma)",
            data=convert_df(df_pitch_class),
            file_name="Pitch_class(chroma).csv",
            mime="text/csv",
        )