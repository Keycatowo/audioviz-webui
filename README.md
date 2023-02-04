# music-analysis
音樂分析工具
## 程式
相關程式來源於[這裡](https://github.com/TrangDuLam/NTHU_Music_AI_Tools)。

## 安裝及執行方式

```sh
# 下載專案
git clone https://github.com/Keycatowo/music-analysis.git

# 安裝相關套件
pip install -r requirements.txt

# 執行
streamlit run home.py
```

## 範例
![](fig/demo.gif)

## 開發日誌
+ 2023/02/04：完成part2功能
+ 2023/02/03：完成part1功能、完成part1使用說明
+ 2023/01/19：新增part6功能草稿、更新README安裝方式
+ 2023/01/03：新增part1, 3功能草稿
+ 2022/12/29：Windows下環境打包
+ 2022/12/20：拆分成多頁面組成結構
+ 2022/12/14：建立專案、新增檔案讀取、檔案基本資訊


## TODO
+ Feature：範例檔案功能
+ BUG：多頁面的拉杆在切換的時候顯示會不同步，但功能正常
+ Feature：包裝成Docker
+ Feature：部署到Vercel demo


## 版本記錄

### v0.1.0 完成Basic Information
![](fig/v0.1.0.gif)