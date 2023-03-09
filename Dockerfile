FROM python:3.8-slim-buster

# 將工作目錄設定為 /app
WORKDIR /app

# 複製當前目錄下的所有檔案到 /app
COPY . /app

# 開啟 8501 port
EXPOSE 8501

# 安裝套件
RUN apt update && \
    apt upgrade -y && \
    apt install -y libsndfile1 && \
    apt install -y ffmpeg

# 升級 pip, 安裝套件
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# 執行命令
CMD ["streamlit", "run", "home.py"]
