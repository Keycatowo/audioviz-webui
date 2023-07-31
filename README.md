# Audioviz - 視覺化音樂分析工具

[![Dockerhub CI](https://github.com/Keycatowo/audioviz-webui/actions/workflows/docker.yml/badge.svg?brach=main&style=plastic)](https://github.com/Keycatowo/audioviz-webui/actions/workflows/docker.yml)
[![Streamlit Cloud CI](https://github.com/Keycatowo/audioviz-webui/actions/workflows/st_cloud.yml/badge.svg?style=plastic)](https://github.com/Keycatowo/audioviz-webui/actions/workflows/st_cloud.yml)
[![Dockerhub Pulls](https://img.shields.io/docker/pulls/owohub/audioviz.svg)](https://hub.docker.com/repository/docker/owohub/audioviz/general)
![Commits](https://img.shields.io/github/commit-activity/m/Keycatowo/music-analysis)
![Views](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FKeycatowo%2Fmusic-analysis&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Views&edge_flat=false)
![GitHub](https://img.shields.io/github/license/Keycatowo/audioviz-webui?style=plastic) 
![GitHub repo size](https://img.shields.io/github/repo-size/Keycatowo/audioviz-webui?style=plastic) 


Audioviz是一個音樂分析工具，提供基本的音樂分析功能，包括音檔基本資訊、音高估計、節奏追蹤、和弦識別、曲式分析和音色分析等功能。

![](fig/demo.gif)

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

## 安裝及執行方式
本工具目前提供以下幾種執行方式：
+ 安裝Docker容器，在容器中執行
+ 若有Python環境，可在Python環境中執行
+ 執行於Streamlit Cloud：不需要安裝Python，但效能會受限於雲端運算資源

幾種方式比較：

安裝方式 | 安裝流程 | 執行效率 | 備註
--- | --- | --- | ---
Docker | ⭐⭐⭐ | ⭐⭐⭐| 不需要安裝Python，且效能較好
Python | ⭐⭐ | ⭐⭐⭐ | 可以自行修改程式碼，但可能需要注意相依套件版本
Streamlit Cloud | ⭐ | ⭐ | 容易上手，但效能較差

### Docker執行
Docker Image: [owohub/audioviz](https://hub.docker.com/r/owohub/audioviz)  
如尚無安裝Docker，步驟詳見[說明文件](https://grace1287986s-organization.gitbook.io/audioviz-ui-v2/installation)

### 本地端執行(Python)
需要有Python環境，並安裝相關套件，執行方式如下：
```sh
# 下載專案
git clone https://github.com/Keycatowo/audioviz-webui.git
cd audioviz-webui

# 安裝相關套件
pip install -r requirements.txt

# 執行
streamlit run home.py
```

### Streamlit Cloud
因為執行記憶體限制，網頁範例有限制檔案大小，無法上傳過大的音樂檔案。

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://audioviz.streamlit.app/)


## 問題反饋
如果您在使用此專案時遇到任何問題，請通過以下方式與我們聯繫：

- 發送電子郵件給我們
- 在我們的GitHub頁面提交問題
    - [視覺化介面](https://github.com/Keycatowo/audioviz-webui/issues)
    - [程式套件](https://github.com/TrangDuLam/audioviz/issues)

我們會盡快回复您的問題。

## 授權協議

音樂分析工具採用 [MIT](https://opensource.org/license/mit/) 授權。

請注意，我們的軟件和內容可能包含第三方軟件庫和組件，這些庫和組件受到其各自的許可證的管轄。有關這些庫和組件的詳細信息，請參閱相應的文檔。







