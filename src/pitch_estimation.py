import librosa
from librosa import display
from librosa import feature

import numpy as np
from matplotlib import pyplot as plt
import scipy

from numpy import typing as npt
from typing import Optional, Tuple, Union


import plotly.graph_objects as go
import seaborn as sns

import pandas as pd


def plot_mel_spectrogram(
        y: npt.ArrayLike, 
        sr:int, 
        shift_array: npt.ArrayLike,
        with_pitch : bool = True,
        ax = None,
        show_colorbar : bool = True,
    ):

    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    if with_pitch :
        
        f0, voiced_flag, voiced_probs = librosa.pyin(y,
                                                     fmin=librosa.note_to_hz('C2'),
                                                     fmax=librosa.note_to_hz('C7'))
        times = librosa.times_like(f0, sr=sr)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12,6))
        else:
            fig = ax.get_figure()
        img = librosa.display.specshow(S_dB, x_axis='time',
                                       y_axis='mel', sr=sr, 
                                       fmax=8000, ax=ax)
        ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
        ax.set_xticks(shift_array - shift_array[0],
                      shift_array)
        if show_colorbar :
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.legend(loc='upper right')
        ax.set(title='Mel-frequency spectrogram')


    else :
        if ax is None:
            fig, ax = plt.subplots(figsize=(12,6))
        else:
            fig = ax.get_figure()
        img = librosa.display.specshow(S_dB, x_axis='time',
                                       y_axis='mel', sr=sr, 
                                       fmax=8000, ax=ax)
        ax.set_xticks(shift_array - shift_array[0],
                      shift_array)
        if show_colorbar:
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')
    ax.set_xlabel('Time (s)')
    
    return fig, ax

def plot_constant_q_transform(y: npt.ArrayLike, sr:int,
                              shift_array: npt.ArrayLike
    ) :

    C = np.abs(librosa.cqt(y, sr=sr))
    fig, ax = plt.subplots(figsize=(12,6))
    img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                                   sr=sr, x_axis='time', y_axis='cqt_note', ax=ax)
    ax.set_xticks(shift_array - shift_array[0],
                      shift_array)
    ax.set_title('Constant-Q power spectrum')
    ax.set_xlabel('Time (s)')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    return fig, ax
    
def plot_chroma(
    y: npt.ArrayLike, 
    sr: int, 
    shift_time: Optional[float] = 0.0, 
    intervals: Optional[int] = 10, 
    return_data: bool = False, 
    use_plotly: bool = False
) -> Union[
    Tuple[plt.Figure, plt.Axes], 
    Tuple[go.Figure, None], 
    Tuple[plt.Figure, plt.Axes, np.ndarray, np.ndarray],
    Tuple[go.Figure, None, np.ndarray, np.ndarray]
]:
    """
    Plot the chromagram of an audio signal using Matplotlib or Plotly.

    Parameters:
    -----------
    y : np.ndarray [shape=(n,)] or None
        Audio time series.
    sr : number > 0 [scalar]
        Sampling rate of y.
    shift_time : float, default=0.0
        Amount of time (in seconds) to shift the horizontal axis of the chromagram.
    intervals : int, default=10
        Number of frames to display on the horizontal axis.
    return_data : bool, default=False
        If True, returns chromagram and times as well as plot.
    use_plotly : bool, default=False
        If True, use Plotly instead of Matplotlib to create plot.

    Returns:
    --------
    fig : matplotlib.figure.Figure object or plotly.graph_objs._figure.Figure object
        The resulting plot.
    ax : matplotlib.axes.Axes object or None
        Axes of the plot if Matplotlib was used, otherwise None.
    chroma : np.ndarray
        Chromagram of the audio signal if return_data=True.
    chroma_times : np.ndarray
        Array of times corresponding to the chromagram if return_data=True.
    """
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_times = librosa.times_like(chroma, sr=sr)
    num_frames = chroma.shape[1]
    # 取出10個frame的index和chroma_t
    selected_frames = np.linspace(0, num_frames-1, intervals, dtype=int)
    selected_frame_times = np.round(chroma_times[selected_frames] + shift_time, 2)
    
    if use_plotly:
        fig = go.Figure(
            data=go.Heatmap(z=chroma,
                            x=chroma_times + shift_time,
                            y=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            )
        )
        fig.update_layout(title="Chroma",
                          xaxis_title="Time(s)",
                          yaxis_title="Pitch Class")
        ax = None
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(chroma, ax=ax)
        ax.set_title("Chroma")
        ax.set_xlabel("Time(s)")
        ax.invert_yaxis()
        ax.set_yticklabels(
            ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
            rotation=0
        )
        ax.set_xticks(selected_frames)
        ax.set_xticklabels(selected_frame_times, rotation=0)
    
    if return_data:
        return fig, ax, chroma, chroma_times
    else:
        return fig, ax
    
def plot_pitch_class(
    y: npt.ArrayLike,           # 音訊資料
    sr: int,                    # 取樣率
    resolution_ratio: int = 1,  # 音高類別解析度倍率，預設為 1
    use_plotly: bool = False,   # 是否使用 Plotly 繪圖，預設為 False
    return_data: bool = False,  # 是否回傳數據，預設為 False
):
    """
    繪製音高類別出現機率的長條圖。此函式會計算音高類別出現機率，
    然後根據使用者的指定來決定使用 Matplotlib 或 Plotly 繪圖，或是回傳數據。

    Args:
        y (npt.ArrayLike): 音訊資料，可以是 ndarray, list, tuple, 或 Tensor
        sr (int): 取樣率，代表每秒樣本數
        resolution_ratio (int, optional): 音高類別解析度倍率，預設為 1
        use_plotly (bool, optional): 是否使用 Plotly 繪圖，預設為 False
        return_data (bool, optional): 是否回傳數據，預設為 False

    Returns:
        tuple: 若 return_data 為 False，回傳 tuple (fig, ax)，
               fig 為繪製的圖片物件，ax 為 Matplotlib 的 Axes 物件。
               若 return_data 為 True，回傳 tuple (fig, ax, df)，
               df 為 Pandas 的 DataFrame 物件，用於儲存音高類別出現機率。

    Raises:
        None
    """
    
    
    # 計算音高類別出現機率
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_colors = ['#636EFA', '#00CC96']
    chroma = librosa.feature.chroma_stft(
        S=np.abs(librosa.stft(y)), 
        sr=sr, 
        n_chroma=12*resolution_ratio
    )
    note_probs = np.mean(chroma >= 1/np.sqrt(2), axis=1)
    note_probs = note_probs / note_probs.sum() # normalize
    # 將note_probs rolling (resolution_ratio/2)格
    note_probs = np.roll(note_probs, resolution_ratio//2)
    
    # 建立位置mask
    unit_pos = np.zeros(resolution_ratio, dtype=bool) # 一個音的標記長度
    unit_pos[resolution_ratio//2] = True # 繪製中間為True
    note_pos_mask = np.tile(unit_pos, 12) # 重複12次

    if use_plotly:
        fig = go.Figure()
        ax = None
        fig.add_trace(
            go.Bar(
                x=list(range(12 * resolution_ratio)), 
                y=note_probs*100, 
                marker_color=np.tile(np.repeat(note_colors, resolution_ratio), 6)
            )
        )
        fig.update_layout(title='Pitch Classes', yaxis_title='Occurrence', yaxis_ticksuffix='%')
        fig.update_layout(
            xaxis= dict(
                tickmode = 'array',
                tickvals = np.arange(12 * resolution_ratio)[note_pos_mask],
                ticktext = note_names,
            )
        )
    else:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(
                    x=list(range(12 * resolution_ratio)), 
                    y=note_probs*100, 
                    hue=np.tile(np.repeat(note_colors, resolution_ratio), 6),
                    dodge=False,
                    ax=ax,
        )
        ax.legend().remove()
        ax.set_title('Pitch Classes')
        ax.set(xlabel='Pitch Classes', ylabel='Occurrence (%)')
        ax.set_xticks(
            np.arange(12*resolution_ratio)[note_pos_mask],
            note_names
        )
    
    if return_data:
        note_array = np.zeros(resolution_ratio * 12, dtype="U3")
        note_array[note_pos_mask] = note_names
        df = pd.DataFrame(
            note_probs, columns=["Prob"]
        ).set_axis(note_array, axis=0)
        return fig, ax, df
    else:
        return fig, ax