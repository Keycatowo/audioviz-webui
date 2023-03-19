## 開發日誌
+ 2023/03/19：截取片段改至側邊控制
+ 2023/03/18：重構Part2-Pitch_class, Part1-waveform, Part1-spectrogram
+ 2023/03/11：新增Dockerhub push CI
+ 2023/03/09：新增Dockerfile
+ 2023/03/08：新增Github Badgets
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

## 版本記錄

### v0.9.0-alpha
+ 修正錯誤
    + 修正User調整Chord圖形無變化的問題(#19)
+ 新增部分操作功能
    + 重構Pitch Class, Waveform, Spectrogram
    + 新增側邊欄位控制截取片段，移除原本的拉杆

### v0.8.0-alpha
+ 修正錯誤
    + 修正不同頁面切換時slider顯示同步問題
    + 修正Part3時間軸位移問題
+ 新增部分操作功能
    + Part4-1 使用者可調整和弦重新繪製
    + 新增Choma的Heatmap和下載
    + 新增Onset和Beats資料下載區塊
    + 新增聲音片段的時間位移會影響到繪圖的功能
    + 新增Part3 Onset和Beat的顯示欄位
+ 文件與風格
    + 更新打包版本安裝方式
    + 統一座標軸標籤時間為Time(s)，除了必須要以frame顯示的部分
    + 移除內建頁面說明、統一增加標題
    + 更新說明介面、包含開發團隊和授權

### v0.7.0-alpha
+ 修正sample rate錯誤
+ 改為tabs結構
+ 新增部分操作功能
    + Part2: Chrome下載csv功能、12/120 Classes圖
    + Part3: Onset與Beats調整區塊
    + Part6：可調整rollof選項、下載Spectrogram的csv

### v0.6.1-alpha 整理發佈alpha版
+ 整理程式與README
+ 發佈alpha版程式

### v0.6.0-pre-alpha 加入Structure Analysis
![](fig/v0.6.0.gif)

### v0.5.0-pre-alpha 加入Timber Analysis
![](fig/v0.5.0.gif)

### v0.4.0-pre-alpha 加入Chord Recognition
![](fig/v0.4.0.gif)

### v0.3.0-pre-alpha 加入Beat Tracking
![](fig/v0.3.0.gif)

### v0.2.0-pre-alpha 加入Pitch Estimation
![](fig/v0.2.0.gif)

### v0.1.0-pre-alpha 加入Basic Information
![](fig/v0.1.0.gif)

### v0.0.1-pre-alpha 建立專案
