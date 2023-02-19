import librosa
from librosa import display
from librosa import feature

import numpy as np
from matplotlib import pyplot as plt
import scipy

from numpy import typing as npt
import typing


def show_duration(y: npt.ArrayLike, sr: int) -> float:
    pass


def selcet_time(start_time: float, end_time: float) :
    pass


def plot_waveform(ax, y: npt.ArrayLike, sr: int, start_time: float = 0.0, end_time: float = None) -> None :
    # ax = plt.subplot(2, 1, 1)
    startIdx = int(start_time * sr)
    
    if not end_time :
        
        librosa.display.waveshow(y[startIdx:], sr)
    
    else :
        endIdx = int(end_time * sr)
        librosa.display.waveshow(y[startIdx:endIdx - 1], sr)   
    
    return


def signal_RMS_analysis(y: npt.ArrayLike, show_plot: bool = True) :

    fig, ax = plt.subplots()

    rms = librosa.feature.rms(y = y)
    times = librosa.times_like(rms)

    if show_plot :
        ax.plot(times, rms[0])

    return fig, ax, times, rms