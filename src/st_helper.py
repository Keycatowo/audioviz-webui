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