#%%
import streamlit as st
import librosa
from src.timbre_analysis import spectral_centroid_analysis, rolloff_frequency_analysis

st.title('Timbre Analysis')

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

    # spectral_centroid_analysis
    st.subheader("spectral_centroid_analysis")
    fig, ax = spectral_centroid_analysis(y, sr)
    st.pyplot(fig)

    # # rolloff_frequency_analysis
    # st.subheader("rolloff_frequency_analysis")
    # fig, ax = rolloff_frequency_analysis(y, sr)
    # st.pyplot(fig)
