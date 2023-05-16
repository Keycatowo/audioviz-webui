#%%
import streamlit as st

st.header("Music Analysis Tool")
st.session_state.debug = False

if "inited" not in st.session_state:
    st.session_state["inited"] = True
    
    st.session_state.start_time = 0.0
    st.session_state.first_run = True
    st.session_state["use_segment"] = False
    st.session_state["use_plotly"] = False
    st.session_state["file_name"] = ""

    st.session_state["0-file"] = {
    }
    st.session_state["1-basic"] = {
        # option
        "use_pitch_name": False,
        
    }
    st.session_state["2-Pitch"] = {
        # option
        "show_f0": True,
        "resolution_ratio": 1
        # data
    }
    st.session_state["3-Time"] = {
        # option
        "onset_frames": [],
        "onset_ma_window": 3,
        
        "onset_method_standard": True,
        "onset_method_mel": False,
        "onset_method_cqt": False,
        
        "beat_frames": [],
        "beat_ma_window": 3,
        
    }
    st.session_state["4-Chord"] = {
        # data
        "chord_df": None,
        
        # flag
        "chord_df_ready": False,
    }
    st.session_state["5-Structure"] = {}
    st.session_state["6-Timbre"] = {}


st.write(
"""
# 音樂分析工具

此工具整合Pitch_estimation、Beat Tracking、Chord recognition、Structure analysis和Timbre analysis等功能，旨在提供一個簡便易用的音樂分析工具。

## 功能概述

以下是此工具的主要功能：

- Basic analysis：音檔基本資訊
- Pitch analysis：樂曲音高估計
- Time analysis：節奏追蹤
- Chord analysis：和弦識別
- Structure analysis：曲式分析
- Timbre analysis：音色分析

我們希望此專案可以幫助不具備程式基礎的音樂工作者和愛好者進行音樂分析，
透過整合現有的各種音樂分析方法與工具，並將其整合在一個簡單易用的網頁工具介面中。

## 開發團隊
+ [Li Su](https://www.iis.sinica.edu.tw/pages/lisu/contact_zh.html)
    + 中央研究院 資訊科學研究所 音樂與文化科技實驗室(Music & Culture Technology Lab, Institute of Information Science, Academia Sinica, Taiwan)
    + 計畫主持人 
+ [Yu-Fen Huang](https://yfhuang.info/)
    + 中央研究院 資訊科學研究所 音樂與文化科技實驗室(Music & Culture Technology Lab, Institute of Information Science, Academia Sinica, Taiwan)
    + 計畫主持人
+ [Yu-Lan Chuang](https://github.com/TrangDuLam)
    + 清華大學 電機工程所
    + 核心功能開發、套件源碼整合
+ [Hong-Hsiang Liu](https://blog.o-w-o.cc)
    + 清華大學 電機工程所
    + 代碼重構、互動介面設計、應用部署與配置
+ Ting-Yi Lu
    + 清華大學 資訊工程所
    + 套件源碼整合、說明文件撰寫

## 工具相關資源
+ [視覺化介面](https://github.com/Keycatowo/music-analysis)：適合不具備程式基礎的使用者
+ [程式套件](https://github.com/TrangDuLam/NTHU_Music_AI_Tools)：提供更多細節調整，適合具備程式基礎的使用者
+ 說明文件：
    + ...

## 問題反饋

如果您在使用此專案時遇到任何問題，請通過以下方式與我們聯繫：

- 發送電子郵件給我們
- 在我們的GitHub頁面提交問題
    - [視覺化介面](https://github.com/Keycatowo/music-analysis/issues)
    - [程式套件](https://github.com/TrangDuLam/NTHU_Music_AI_Tools)

我們會盡快回复您的問題。

## 授權協議

音樂分析工具採用 [MIT](https://opensource.org/license/mit/) 授權。

請注意，我們的軟件和內容可能包含第三方軟件庫和組件，這些庫和組件受到其各自的許可證的管轄。有關這些庫和組件的詳細信息，請參閱相應的文檔。
"""
)