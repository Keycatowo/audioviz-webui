import streamlit as st

@st.experimental_memo
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
            
            
def get_shift(start_time, end_time):
    """
        回傳從start_time到end_time的時間刻度
        開頭為start_time，結尾為end_time
        中間每隔1秒一個刻度
        
        return: a np.array of time stamps
    """
    import numpy as np
    
    shift_array = np.arange(start_time, end_time, 1)
    if shift_array[-1] != end_time:
        shift_array = np.append(shift_array, end_time)
    
    shift_array = np.round(shift_array, 1)
    return start_time, shift_array
    