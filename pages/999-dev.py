#%%
import pkg_resources
import streamlit as st

with st.expander("Show packages"):
    for dist in pkg_resources.working_set:
        print(f"{dist.project_name}=={dist.version}")
        st.write(f"{dist.project_name}=={dist.version}")

#%%
import os
import psutil

with st.expander("Show memory usage"):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
    st.write(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
    
    
st.session_state["use_plotly"] = st.checkbox("Use plotly", value=st.session_state["use_plotly"])
st.session_state["debug"] = st.checkbox("Debug", value=st.session_state["debug"])



with st.expander("Session state"):
    st.write(type(st.session_state))
    st.write(st.session_state)
    session_state = dict(st.session_state)
    st.write(type(session_state))
    st.write(session_state)
    

import pickle




# from src.st_helper import download_upload_settings

# download_upload_settings()