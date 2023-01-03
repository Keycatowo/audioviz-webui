#%%
import streamlit as st
import librosa
from src.beat_track import beat_analysis, static_tempo_estimation, onset_and_beat_analysis

st.title('Beat Tracking')

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

    # Beat analysis
    st.subheader("Beat analysis")
    fig, ax = beat_analysis(y, sr)
    st.pyplot(fig)

    # Static tempo estimation
    st.subheader("Static tempo estimation")
    fig, ax = static_tempo_estimation(y, sr)
    st.pyplot(fig)

