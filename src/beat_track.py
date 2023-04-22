import librosa
from librosa import display
from librosa import feature

import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import scipy
import soundfile as sf

from numpy import typing as npt
from typing import List, Tuple

def onsets_detection(y: npt.ArrayLike, sr: int, shift_array: npt.ArrayLike) -> tuple :
    """
        計算音檔的onset frames
    """
    o_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(o_env, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    D = np.abs(librosa.stft(y))

    fig, ax = plt.subplots()
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                             x_axis='time', y_axis='log', ax=ax, sr=sr)
    ax.set_xticks(shift_array - shift_array[0],
                      shift_array)
    ax.set_xlabel('Time (s)')
    ax.autoscale()
    ax.set(title='Power spectrogram')


    return fig, ax, (o_env, times, onset_frames)

def onset_click_plot(
    o_env, 
    times, 
    onset_frames, 
    y_len, 
    sr, 
    shift_time,
    ax = None
) -> tuple:
    """
        重新繪製onset frames
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.plot(times + shift_time, o_env, label='Onset strength')
    ax.vlines(times[onset_frames] + shift_time, 0, o_env.max(), color='r', alpha=0.9,
              linestyles='--', label='Onsets')
    ax.autoscale()
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Strength')
    ax.set_xlim([shift_time, shift_time + y_len / sr])
    
    y_onset_clicks = librosa.clicks(frames=onset_frames, sr=sr, length=y_len)
    return fig, ax, y_onset_clicks
    

def plot_onset_strength(y: npt.ArrayLike, sr:int, standard: bool = True, custom_mel: bool = False, cqt: bool = False, shift_array: npt.ArrayLike = None) -> tuple:
    
    D = np.abs(librosa.stft(y))
    times = librosa.times_like(D, sr)

    fig, ax = plt.subplots(nrows=2, sharex=True)
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax[0], sr=sr)
    
    ax[0].set(title='Power spectrogram')
    ax[0].label_outer()

    # Standard Onset Fuction 

    if standard :
        onset_env_standard = librosa.onset.onset_strength(y=y, sr=sr)
        ax[1].plot(times, 2 + onset_env_standard / onset_env_standard.max(), alpha=0.8, label='Mean (mel)')
    
    if custom_mel :
        onset_env_mel = librosa.onset.onset_strength(y=y, sr=sr,
                                                     aggregate=np.median,
                                                     fmax=8000, n_mels=256)
        ax[1].plot(times, 1 + onset_env_mel / onset_env_mel.max(), alpha=0.8, label='Median (custom mel)')
    
    if cqt :
        C = np.abs(librosa.cqt(y=y, sr=sr))
        onset_env_cqt = librosa.onset.onset_strength(sr=sr, S=librosa.amplitude_to_db(C, ref=np.max))
        ax[1].plot(times, onset_env_cqt / onset_env_cqt.max(), alpha=0.8, label='Mean (CQT)')

    ax[1].legend()
    ax[1].set(ylabel='Normalized strength', yticks=[])
    ax[1].set_xticks(shift_array - shift_array[0],
                         shift_array)
    ax[1].autoscale()
    ax[1].set_xlabel('Time (s)')

    return fig, ax


def beat_analysis(y: npt.ArrayLike, sr:int, spec_type: str = 'mel', spec_hop_length: int = 512, shift_array: npt.ArrayLike = None, ax=None) :
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    times = librosa.times_like(onset_env, sr=sr, hop_length=spec_hop_length)

    if spec_type == 'mel':
        M = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=spec_hop_length)
        librosa.display.specshow(librosa.power_to_db(M, ref=np.max), 
                                 y_axis='mel', x_axis='time', hop_length=spec_hop_length,
                                 ax=ax, sr=sr)
        ax.set(title='Mel spectrogram')

    if spec_type == 'stft':
        S = np.abs(librosa.stft(y))
        img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), 
                                       y_axis='log', x_axis='time', ax=ax, sr=sr)
        
        ax.set_title('Power spectrogram')
        # fig.colorbar(img, ax=ax[0], format="%+2.0f dB")
    if shift_array:
        ax.set_xticks(shift_array - shift_array[0],
                      shift_array)
    ax.autoscale()
    ax.set_xlabel('Time (s)')
    
    
    return fig, ax, (times, onset_env, tempo, beats)

def beat_plot(times, onset_env, tempo, beats, y_len, sr, shift_time, ax=None):
    """
        重新繪製beat
    """
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.plot(times + shift_time, librosa.util.normalize(onset_env), label='Beat strength')
    ax.vlines(times[beats] + shift_time, 0, 1, alpha=0.5, color='r', linestyle='--', label='Beats')
    tempoString = 'Tempo = %.2f'% (tempo)
    ax.plot([], [], ' ', label = tempoString)
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized strength')
    
    y_beats = librosa.clicks(frames=beats, sr=sr, length=y_len)
    
    return fig, ax, y_beats

def predominant_local_pulse(y: npt.ArrayLike, sr:int, shift_time:float=0) -> tuple :

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
    times = librosa.times_like(pulse, sr=sr)

    fig, ax = plt.subplots()
    ax.plot(times + shift_time, librosa.util.normalize(pulse),label='PLP')
    ax.vlines(times[beats_plp] + shift_time, 0, 1, alpha=0.5, color='r', 
             linestyle='--', label='PLP Beats')
    ax.legend()
    ax.set(title="Predominant local pulse")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized strength')

    return fig, ax


def static_tempo_estimation(y: npt.ArrayLike, sr: int, hop_length: int = 512) -> tuple:
  
  '''
  To visualize the result of static tempo estimation
  
  y: input signal array
  sr: sampling rate
  
  '''

  onset_env = librosa.onset.onset_strength(y=y, sr=sr)
  tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

  # Static tempo estimation
  prior = scipy.stats.uniform(30, 300)  # uniform over 30-300 BPM
  utempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, prior=prior)

  tempo = tempo.item()
  utempo = utempo.item()
  ac = librosa.autocorrelate(onset_env, max_size=2 * sr // hop_length)
  freqs = librosa.tempo_frequencies(len(ac), sr=sr,
                                   hop_length=hop_length)

  fig, ax = plt.subplots()
  ax.semilogx(freqs[1:], librosa.util.normalize(ac)[1:],
              label='Onset autocorrelation', base=2)
  ax.axvline(tempo, 0, 1, alpha=0.75, linestyle='--', color='r',
             label='Tempo (default prior): {:.2f} BPM'.format(tempo))    
  ax.axvline(utempo, 0, 1, alpha=0.75, linestyle=':', color='g',
             label='Tempo (uniform prior): {:.2f} BPM'.format(utempo)) 
  ax.set(xlabel='Tempo (BPM)', title='Static tempo estimation')
  ax.grid(True)
  ax.legend() 

  return fig, ax


def plot_tempogram(y: npt.ArrayLike, sr: int, type: str = 'autocorr', hop_length: int = 512, shift_array: npt.ArrayLike = None) -> tuple :
    
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length)[0]

    fig, ax = plt.subplots()

    if type == 'fourier' :
        # To determine which temp to show?
        librosa.display.specshow(np.abs(tempogram), sr=sr, hop_length=hop_length, 
                                 x_axis='time', y_axis='fourier_tempo', cmap='magma')
        ax.axhline(tempo, color='w', linestyle='--', alpha=1, label='Estimated tempo={:g}'.format(tempo))
        ax.legend(loc='upper right')
        # ax.title('Fourier Tempogram')

    if type == 'autocorr' :
        ac_tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length, norm=None)
        librosa.display.specshow(ac_tempogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='tempo', cmap='magma')
        ax.axhline(tempo, color='w', linestyle='--', alpha=1, label='Estimated tempo={:g}'.format(tempo))
        ax.legend(loc='upper right')
        # ax.title('Autocorrelation Tempogram')
    ax.set_xticks(shift_array - shift_array[0],
                      shift_array)
    ax.autoscale()
    
    return fig, ax

def plot_bpm(
    beat_times: List[float], 
    shift_time: float = 0, 
    window_size: int = 1, 
    use_plotly: bool = False,
    ax = None,
    title="Beat Rate Curve",
    xtitle = "Time (s)",
    ytitle = "Beats / min",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Parameters:
        beat_times (List[float]): 節拍時間的數組。
        time_shift (float): 將時間軸上的點向右移動的時間量。
        window_size (int): 用於計算移動平均數的窗口大小。
        plot_with_plotly (bool): 如果為 True，使用 Plotly 繪製曲線；否則，使用 Matplotlib 繪製曲線。

    Returns:
        Tuple[plt.Figure, plt.Axes] or go.Figure: 返回繪製的圖形對象。
        如果 `plot_with_plotly` 為 True，返回 go.Figure 對象；否則，返回 plt.Figure 和 plt.Axes 對象。

    Raises:
        ValueError: 如果 `beat_times` 不是一個有效的數字數組。
        ValueError: 如果 `window_size` 不是正整數。
    """

    times_diff = np.diff(beat_times)
    times_diff_ma = np.convolve(times_diff, np.ones(window_size)/window_size, mode='same')
    rate = 1/times_diff_ma * 60
    
    if use_plotly:
        fig = go.Figure(data=go.Scatter(x=beat_times[:-1] + shift_time, y=rate, mode='lines+markers', name=f'BPM (MA{window_size})'))
        ax = None
        fig.update_layout(
            title=title, 
            xaxis_title=xtitle, 
            yaxis_title=ytitle,
            showlegend=True,
        )    
        
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.get_figure()
        ax.plot(beat_times[:-1] + shift_time, rate, label=f'BPM (MA{window_size})')
        ax.set_ylim(0, 280)
        ax.set_title(title)
        ax.set_xlabel(xtitle)
        ax.set_ylabel(ytitle)
        ax.legend()
    
    return fig, ax