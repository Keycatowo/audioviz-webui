#%%
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd
from src.beat_track import onsets_detection, plot_onset_strength, beat_analysis, predominant_local_pulse, static_tempo_estimation, plot_tempogram, onset_click_plot, beat_plot, plot_bpm
from src.st_helper import convert_df, get_shift, update_sessions, use_plotly, sengment_change_clean
import numpy as np

st.title('Time Analysis')
#%% 除錯訊息
if st.session_state.debug:
    st.write(st.session_state)

#%% 上傳檔案區塊
with st.expander("上傳檔案(Upload Files)"):
    file = st.file_uploader("Upload your music library", type=["mp3", "wav", "ogg"])

    if file is not None:
        st.audio(file, format="audio/ogg")
        st.subheader("File information")
        st.write(f"File name: `{file.name}`", )
        st.session_state["file_name"] = file.name
        st.write(f"File type: `{file.type}`")
        st.write(f"File size: `{file.size}`")

        # 載入音檔
        y, sr = librosa.load(file, sr=22050)
        st.write(f"Sample rate: `{sr}`")
        duration = float(np.round(len(y)/sr-0.005, 2)) # 時間長度，取小數點後2位，向下取整避免超過音檔長度
        st.write(f"Duration(s): `{duration}`")
        y_all = y
        start_time = 0
        end_time = duration
 
#%% 更新session
update_sessions()       
#%% 片段模式
if file is not None:
    use_segment = st.sidebar.checkbox("使用片段模式", value=st.session_state["use_segment"], key="segment")
    st.session_state["use_segment"] = use_segment
    
    if use_segment:
        # 若使用片段模式，則顯示選擇片段的起始時間與結束時間
        if st.session_state.first_run:
            start_time = st.sidebar.number_input("開始時間", value=0.0, min_value=0.0, max_value=duration, step=0.01, on_change=sengment_change_clean)
            end_time = st.sidebar.number_input("結束時間", value=duration, min_value=0.0, max_value=duration, step=0.01, on_change=sengment_change_clean)
            st.session_state.first_run = False
            st.session_state.start_time = start_time
            st.session_state.end_time = end_time
        else:
            start_time = st.sidebar.number_input("開始時間", value=st.session_state.start_time, min_value=0.0, max_value=duration, step=0.01, on_change=sengment_change_clean)
            end_time = st.sidebar.number_input("結束時間", value=st.session_state.end_time, min_value=0.0, max_value=duration, step=0.01, on_change=sengment_change_clean)
            st.session_state.start_time = start_time
            st.session_state.end_time = end_time

    else:
        start_time = 0.0
        end_index = duration
    # 根據選擇的聲音片段，取出聲音資料
    start_index = int(start_time*sr)
    end_index = int(end_time*sr)
    if end_index <= start_index:
        st.sidebar.warning("結束時間必須大於開始時間", icon="⚠️")

    y_sub = y_all[start_index:end_index]
    x_sub = np.arange(len(y_sub))/sr
    
    if use_segment: 
        with st.expander("聲音片段(Segment of the audio)"):
            st.write(f"Selected segment: `{start_time}` ~ `{end_time}`, duration: `{end_time-start_time}`")
            st.audio(y_sub, format="audio/ogg", sample_rate=sr)
            
    use_plotly()

#%% 功能區塊
if file is not None:
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "note_detection",
        "onset_strength",
        "beat_analysis",
        "predominant_local_pulse",
        "static_tempo_estimation",
        "Tempogram"])

    shift_time, shift_array = get_shift(start_time, end_time) # shift_array為y_sub的時間刻度

    # onsets_detection
    with tab1:
        st.subheader("Note Detection")
        st.markdown("#### Spectrogram")
        # 計算onset
        fig3_1a, ax3_1a, onset_data = onsets_detection(y_sub, sr, shift_array)
        o_env, o_times, onset_frames = onset_data
        st.pyplot(fig3_1a)
        st.markdown("#### Modify onset")
        # 上傳onset data
        onset_file = st.file_uploader("Upload onset data from Tony", type=["csv"])
        if onset_file is not None:
            onset_data = pd.read_csv(onset_file, header=None, names=["onset", "note", "offset"])
            if st.session_state["debug"]:
                st.dataframe(onset_data, use_container_width=True)
            onset_times = onset_data["onset"].to_numpy()
            # 取得每一個onset_times在o_times中最接近的index
            onset_frames = []
            for onset_time in onset_times:
                onset_frames.append(np.argmin(np.abs(o_times-onset_time)))
            st.session_state["3-Time"]["onset_frames"] = onset_frames
        # 調整onset frame
        if st.session_state["3-Time"]["onset_frames"] == []:
            st.session_state["3-Time"]["onset_frames"] = list(onset_frames)
        clicks = st.multiselect("Modify the onset manually", 
                                list(range(len(o_env))), st.session_state["3-Time"]["onset_frames"])
        st.session_state["3-Time"]["onset_frames"] = clicks
        
        fig3_1b, ax3_1b, y_onset_clicks = onset_click_plot(o_env, o_times, clicks, len(y_sub), sr, shift_time)
        st.pyplot(fig3_1b)
        # 計算bpm
        st.markdown("#### Onset Ratio Curve")
        onset_beat_window = st.slider("Window Size", 
                                        min_value=1, max_value=10, 
                                        value=st.session_state["3-Time"]["onset_ma_window"],  # default: 3
                                        step=1,
                                        key="onset_beat_window",
                                        help="If the size is 3, the onset ratio curve will be the average of every 3 seconds."
                                    )
        st.session_state["3-Time"]["onset_ma_window"] = onset_beat_window
        if st.session_state["use_plotly"]:
            fig3_1c, ax3_1c = plot_bpm(o_times[clicks], shift_time, onset_beat_window, True, title="Onset Ratio Curve", ytitle="Onsets per minute")
            st.plotly_chart(fig3_1c)
        else:
            fig3_1c, ax3_1c = plot_bpm(o_times[clicks], shift_time, onset_beat_window, False, title="Onset Ratio Curve", ytitle="Onsets per minute")
            st.pyplot(fig3_1c)
        # 下載onset data
        st.markdown("#### Onset Data")
        df_onset = pd.DataFrame({"Frame": clicks, "Time(s)": o_times[clicks], "Onset": o_env[clicks]})
        st.dataframe(df_onset, use_container_width=True)
        st.download_button(
            label="Download onset data",
            data=convert_df(df_onset),
            file_name="onset_data.csv",
        )
            
            
        
        # 播放onset click
        st.markdown("#### Onset Click Preview")
        onset_remix_ratio = st.slider("Onset Click Volume Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        y_onset_remix = y_onset_clicks*onset_remix_ratio + y_sub*(1-onset_remix_ratio)
        y_onset_remix = y_onset_remix/np.max(np.abs(y_onset_remix))
        st.audio(y_onset_remix, format="audio/ogg", sample_rate=sr)
        

    # onset_strength
    with tab2:
        st.subheader("onset_strength")
        onset_strength_standard = st.checkbox("standard", 
                                              value=st.session_state["3-Time"]["onset_method_standard"]) # default: True
        st.session_state["3-Time"]["onset_method_standard"] = onset_strength_standard
        onset_strength_mel = st.checkbox("mel", 
                                                value=st.session_state["3-Time"]["onset_method_mel"]) # default: False
        st.session_state["3-Time"]["onset_method_mel"] = onset_strength_mel
        onset_strength_cqt = st.checkbox("cqt", 
                                         value=st.session_state["3-Time"]["onset_method_cqt"]) # default: False
        st.session_state["3-Time"]["onset_method_cqt"] = onset_strength_cqt
        fig3_2, ax3_2 = plot_onset_strength(y_sub, sr,
            standard=onset_strength_standard,
            custom_mel=onset_strength_mel,
            cqt=onset_strength_cqt,
            shift_array=shift_array
        )
        st.pyplot(fig3_2)

    # beat_analysis
    with tab3:
        st.subheader("beat_analysis")
        st.markdown("#### Spectrogram")
        # 計算beat
        spec_type = st.selectbox("spec_type", ["mel", "stft"])
        spec_hop_length = st.number_input("spec_hop_length", value=512)
        fig3_3a, ax3_3b, beats_data = beat_analysis(y_sub, sr,
            spec_type=spec_type,
            spec_hop_length=spec_hop_length,
            shift_array=shift_array
        )
        b_times, b_env, b_tempo, b_beats = beats_data
        st.pyplot(fig3_3a)
        # 調整beat frame
        st.markdown("#### Modify Beat Clicks")
        if st.session_state["3-Time"]["beat_frames"] == []:
            st.session_state["3-Time"]["beat_frames"] = list(b_beats)
        b_clicks = st.multiselect("Modify the beat manually",
                                  list(range(len(b_env))), 
                                  st.session_state["3-Time"]["beat_frames"]
        )
        st.session_state["3-Time"]["beat_frames"] = b_clicks
        fig3_3b, ax3_3b, y_beat_clicks = beat_plot(b_times, b_env, b_tempo, b_clicks, len(y_sub), sr, shift_time)
        st.pyplot(fig3_3b)
        # 計算bpm
        st.markdown("#### Beat Ratio Curve")
        beat_window = st.slider("Window Size", 
                                min_value=1, max_value=10, 
                                value=st.session_state["3-Time"]["beat_ma_window"], 
                                step=1,
                                key="beat_ma_window",
                                help="If the size is 3, the beat ratio curve will be the average of every 3 seconds."
        )
        st.session_state["3-Time"]["beat_ma_window"] = beat_window
        if st.session_state["use_plotly"]:
            fig3_3c, ax3_3c = plot_bpm(b_times[b_clicks], shift_time, beat_window, True,  ytitle="Beats per minute")
            st.plotly_chart(fig3_3c)
        else:
            fig3_3c, ax3_3c = plot_bpm(b_times[b_clicks], shift_time, beat_window, False,  ytitle="Beats per minute")
            st.pyplot(fig3_3c)
        # 下載beat data
        st.markdown("#### Beat Data")
        df_beats = pd.DataFrame({"Frame": b_clicks, "Time(s)": b_times[b_clicks] + shift_time, "Beats": b_env[b_clicks]})
        st.dataframe(df_beats, use_container_width=True)
        st.download_button(
            label="Download beats data",
            data=convert_df(df_beats),
            file_name="beats_data.csv",
        )
    
        st.markdown("#### Beat Click Preview")
        beat_remix_ratio = st.slider("Beat Click Volume Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        y_beat_remix = y_beat_clicks*beat_remix_ratio + y_sub*(1-beat_remix_ratio)
        y_beat_remix = y_beat_remix/np.max(np.abs(y_beat_remix))
        st.audio(y_beat_remix, format="audio/ogg", sample_rate=sr)


    # predominant_local_pulse
    with tab4:
        st.subheader("predominant_local_pulse")
        fig3_4, ax3_4 = predominant_local_pulse(y_sub, sr, shift_time)
        st.pyplot(fig3_4)

    # static_tempo_estimation
    with tab5:
        st.subheader("static_tempo_estimation")
        static_tempo_estimation_hop_length = st.number_input("hop_length", value=512)
        fig3_5, ax3_5 = static_tempo_estimation(y_sub, sr,
            hop_length=static_tempo_estimation_hop_length
        )
        st.pyplot(fig3_5)

    # Tempogram
    with tab6:
        st.subheader("Tempogram")
        tempogram_type = st.selectbox("tempogram_type", ["fourier", "autocorr"], index=1)
        tempogram_hop_length = st.number_input("Tempogram_hop_length", value=512)
        fig3_6, ax3_6 = plot_tempogram(y_sub, sr,
            type=tempogram_type,
            hop_length=tempogram_hop_length,
            shift_array=shift_array
        )
        st.pyplot(fig3_6)