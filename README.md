# 音樂分析工具 music-analysis tool

![Commits](https://img.shields.io/github/commit-activity/m/Keycatowo/music-analysis) ![Views](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FKeycatowo%2Fmusic-analysis&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Views&edge_flat=false) ![GitHub](https://img.shields.io/github/license/Keycatowo/music-analysis?style=plastic) ![GitHub repo size](https://img.shields.io/github/repo-size/Keycatowo/music-analysis?style=plastic) ![GitHub Release Date - Published_At](https://img.shields.io/github/release-date/Keycatowo/music-analysis) ![GitHub release (latest by date)](https://img.shields.io/github/v/release/Keycatowo/music-analysis) ![GitHub all releases](https://img.shields.io/github/downloads/Keycatowo/music-analysis/total) ![Dockerhub CI](https://github.com/Keycatowo/music-analysis/actions/workflows/docker.yml/badge.svg)



此工具整合Pitch_estimation、Beat Tracking、Chord recognition、Structure analysis和Timbre analysis等功能，旨在提供一個簡便易用的音樂分析工具。


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
本工具目前提供4種執行方式，分別為：
+ 使用docker安裝運行
+ 使用python本地執行：需要有Python環境，執行效率較高
+ 使用Windows免安裝版本：不需要安裝Python，但執行效率較低
+ 使用網頁範例：不需要安裝Python，受限於記憶體大小，無法上傳過大的音樂檔案

### Docker執行
Docker Image: [owohub/audioviz](https://hub.docker.com/r/owohub/audioviz)  
如尚無安裝Docker，步驟詳見[說明文件](https://grace1287986s-organization.gitbook.io/audioviz-ui-v2/installation)

### 本地端執行(Python)
需要有Python環境，並安裝相關套件，執行方式如下：
```sh
# 下載專案
git clone https://github.com/Keycatowo/music-analysis.git
cd music-analysis

# 安裝相關套件
pip install -r requirements.txt

# 執行
streamlit run home.py --server.maxUploadSize 100
```
+ 如果需要更改上傳檔案大小限制，請修改`--server.maxUploadSize`參數，單位為MB

### 本地端執行(Windows免安裝)
如本地無法安裝Python，可使用包含Python的打包版本，執行方式如下：
+ 至[Release](https://github.com/Keycatowo/music-analysis/releases)下載
+ 解壓縮到任意位置，進入資料夾
+ 執行`run.bat`
+ 在跳出視窗中輸入`streamlit run home.py`，並按下Enter
+ 如果沒有自動開啟網頁，請至瀏覽器輸入`http://localhost:8501/`

### 網頁範例
因為執行記憶體限制，網頁範例有限制檔案大小，無法上傳過大的音樂檔案。

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nthu-music-tools.streamlit.app/)


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







