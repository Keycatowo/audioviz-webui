import streamlit as st
import pickle

@st.cache_data
def convert_df(df):
    """
        Convert a pandas dataframe into a csv file.
        For the download button in streamlit.
    """
    return df.to_csv(index=False).encode('utf-8')

def show_readme(filename):
    with st.expander("頁面說明(Page Description)"):
        with open(filename, "r", encoding="utf-8") as f:
            st.markdown(f.read())
            
            
def get_shift(start_time, end_time, step=5):
    """
        回傳從start_time到end_time的時間刻度
        開頭為start_time，結尾為end_time
        中間每隔step秒一個刻度
        
        return: a np.array of time stamps
    """
    import numpy as np
    step = 1 if step < 1 else step
    
    shift_array = np.arange(start_time, end_time, step)
    if shift_array[-1] +step > end_time:
        shift_array = np.append(shift_array[:-1], end_time)
    
    shift_array = np.round(shift_array, 1)
    return start_time, shift_array
    
    
def update_sessions():
    """
        Update the session state.
        Download/Upload button
    """
    if st.session_state["file_name"]:
        
        with st.expander("Config Download/Upload"):
            
            col1, col2 = st.columns([2, 6])

            col1.download_button(label="Download Current Settings",
                                data=pickle.dumps(dict(st.session_state)),
                                file_name=st.session_state["file_name"]+"_config.pkl",
                                help="Click to Download Current Settings")

            session_file = col2.file_uploader(label="Upload Previous Settings",
                                            help="Upload a previously saved settings file"
            )

            if session_file is not None:
                new_session = pickle.loads(session_file.read())
                st.write(new_session)
                st.session_state.update(new_session)


def warning_region(text="This is a warning"):
    """
        A warning region.
        If the user does not select a file, the warning will be shown.
    """
    st.warning(text, icon="⚠️")
    
    
def use_plotly():
    st.session_state["use_plotly"] = st.sidebar.checkbox("Enable Dynamic Graphics", value=st.session_state["use_plotly"])
    if st.session_state["use_plotly"]:
        st.sidebar.info("Dynamic Graphics Enabled, more memory needed.\nWe will use Plotly to draw the figure if it is supported.")
    
    
def sengment_change_clean():
    """
        當模式切換時，清除之前的資料
    """
    
    st.session_state["3-Time"]["onset_frames"] = []
    st.session_state["3-Time"]["beat_frames"] = []