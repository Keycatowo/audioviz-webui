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
from src.chord_recognition import (
    plot_chord_recognition,
    plot_binary_template_chord_recognition,
    chord_table,
    compute_chromagram,
    chord_recognition_template,
    plot_chord,
    plot_user_chord
)

st.title("Chord Analysis")
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

    tab1, tab2, tab3, tab4 = st.tabs(["STFT Chroma", "Chords Result (Default)", "Chords Result (User)", "dev"])
    shift_time, shift_array = get_shift(start_time, end_time) # shift_array為y_sub的時間刻度
    
    # STFT Chroma 
    with tab1:
        chroma, _, _, _, duration = compute_chromagram(y_sub, sr)
        fig4_1, ax4_1 = plot_chord(chroma, "STFT Chroma")
        st.pyplot(fig4_1)
        
    with tab2:
        _, chord_max = chord_recognition_template(chroma, norm_sim='max')
        fig4_2, ax4_2 = plot_chord(chord_max, "Chord Recognition Result", cmap="crest", include_minor=True)
        st.pyplot(fig4_2)
        sec_per_frame = duration/chroma.shape[1]
        chord_results_df = pd.DataFrame({
            "Frame": np.arange(chroma.shape[1]),
            "Time(s)": np.arange(chroma.shape[1])*sec_per_frame + shift_time,
            "Chord": chord_table(chord_max)
        })
        if st.session_state["4-Chord"]["chord_df_ready"] == False:
            st.session_state["4-Chord"]["chord_df"] = chord_results_df
            st.session_state["4-Chord"]["chord_df_ready"] = True
    
    with tab3:
        # 建立chord result dataframe
        if st.button("Reset"):
            st.session_state["4-Chord"]["chord_df"] = chord_results_df.copy()
            chord_user_df = st.session_state["4-Chord"]["chord_df"]
        
        chord_user_df = st.session_state["4-Chord"]["chord_df"]
        
        chord_user_df = st.experimental_data_editor(
            chord_user_df,
            use_container_width=True
        )
        
        fig4_1b, ax4_1b = plot_user_chord(chord_user_df)
        st.pyplot(fig4_1b)

    # plot_binary_template_chord_recognition
    with tab4:
        st.subheader("plot_binary_template_chord_recognition")
        fig4_4, ax4_4 = plot_binary_template_chord_recognition(y_sub, sr)
        st.pyplot(fig4_4)


