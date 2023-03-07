# 音樂分析工具 music-analysis tool

此工具整合Pitch_estimation、Beat Tracking、Chord recognition、Structure analysis和Timbre analysis等功能，旨在提供一個簡便易用的音樂分析工具。

![](fig/demo.gif)

## 功能概述

以下是此工具的主要功能：

- Basic analysis：音檔基本資訊
- Pitch estimation：樂曲音高估計
- Beat Tracking：節奏追蹤
- Chord recognition：和弦識別
- Structure analysis：曲式分析
- Timbre analysis：音色分析

我們希望此專案可以幫助不具備程式基礎的音樂工作者和愛好者進行音樂分析，
透過整合現有的各種音樂分析方法與工具，並將其整合在一個簡單易用的網頁工具介面中。

## 安裝及執行方式
本工具目前提供3種執行方式，分別為：
+ 使用python本地執行：需要有Python環境，執行效率較高
+ 使用Windows免安裝版本：不需要安裝Python，但執行效率較低
+ 使用網頁範例：不需要安裝Python，受限於記憶體大小，無法上傳過大的音樂檔案

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


## 開發日誌
+ 2023/03/05：Part4可允許使用者調整手動辨識的結果、開發人員頁面(套件列表、內存佔用)
+ 2023/03/04：修正多頁面拉杆同步問題、新增區塊下載功能與說明
+ 2023/02/19：修正功能錯誤、改為tabs結構、新增部分操作功能
    + Part2: Chrome下載csv功能、12/120 Classes圖
    + Part3: Onset與Beats調整區塊
    + Part6：可調整rollof選項、下載Spectrogram的csv
+ 2023/02/04：完成part2、part3、part4、part6功能
+ 2023/02/03：完成part1功能、完成part1使用說明
+ 2023/01/19：新增part6功能草稿、更新README安裝方式
+ 2023/01/03：新增part1, 3功能草稿
+ 2022/12/29：Windows下環境打包
+ 2022/12/20：拆分成多頁面組成結構
+ 2022/12/14：建立專案、新增檔案讀取、檔案基本資訊




