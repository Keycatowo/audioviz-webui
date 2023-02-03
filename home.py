#%%
import streamlit as st

st.header("Music Analysis Tool")
st.subheader("This tool will help you analyze your music library")


# show README.md
with open("README.md", "r", encoding="utf-8") as f:
    st.markdown(f.read())



