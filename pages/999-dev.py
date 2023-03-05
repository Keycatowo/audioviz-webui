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
