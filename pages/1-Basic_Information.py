#%%
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd
from src.st_helper import convert_df, show_readme, get_shift
from src.basic_info import plot_waveform, signal_RMS_analysis, plot_spectrogram


st.title("Basic Information")
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
        y, sr = librosa.load(file, sr=44100)
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
    


#%% 功能分頁
if file is not None:

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
        fig1_1, _ = plot_waveform(x_sub, y_sub, shift_time=shift_time, use_plotly=False)
        st.pyplot(fig1_1)
    
    # 繪製聲音波形圖
    with tab2:
        st.subheader("Waveform(plotly)")
        fig1_2, _ = plot_waveform(x_sub, y_sub, shift_time=shift_time, use_plotly=True) 
        st.plotly_chart(fig1_2)

    # 繪製聲音RMS圖
    with tab3:
        st.subheader("signal_RMS_analysis")
        fig1_3, ax1_3, times, rms = signal_RMS_analysis(y_sub, shift_time=shift_time)
        st.pyplot(fig1_3)   

    # 繪製聲音Spectrogram圖(使用librosa繪製)
    with tab4:
        st.subheader("Spectrogram")
        fig1_4, _ = plot_spectrogram(y_sub, sr, shift_time=shift_time, use_plotly=False, shift_array=shift_array)
        st.pyplot(fig1_4)

    # 下載RMS資料
    with tab5:
        st.subheader("Download RMS data")
        
        col1, col2 = st.columns(2)
        with col1:
            rms_df = pd.DataFrame({"Time(s)": times, "RMS": rms[0,:]})
            st.dataframe(rms_df, use_container_width=True)
        with col2:
            st.download_button(
                "Doanload RMS data",
                convert_df(rms_df),
                "rms.csv",
                "text/csv",
                key="download-csv"
            )
        
# %%
