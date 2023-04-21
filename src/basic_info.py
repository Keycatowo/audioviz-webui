import librosa
from librosa import display
from librosa import feature

import numpy as np
from matplotlib import pyplot as plt
import scipy

from numpy import typing as npt
import typing

import plotly.graph_objects as go
import streamlit as st 

def plot_waveform(
    x: npt.ArrayLike, 
    y: npt.ArrayLike, 
    shift_time: float = 0.0, 
    use_plotly=False,
    ax=None,
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """
    Plots a waveform graph.

    Parameters
    ----------
    x : array-like
        The x-coordinates of the waveform data.
    y : array-like
        The y-coordinates of the waveform data.
    shift_time : float, optional
        A time shift to apply to the waveform, in seconds (default 0.0).
    use_plotly : bool, optional
        Whether to use Plotly to plot the waveform (default False).

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objs._figure.Figure
        The generated figure object.
    ax : matplotlib.axes.Axes or None
        The generated axes object. If `use_plotly` is True, this value will be None.
    """
    if use_plotly:
        fig = go.Figure(data=go.Scatter(x=x + shift_time, y=y))
        ax = None
        fig.update_layout(
            title="Waveform",
            xaxis_title="Time(s)",
            yaxis_title="Amplitude",
        )
    else:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        ax.plot(x + shift_time, y)
        ax.set_xlabel("Time(s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Waveform")
        ax.set_xlim([x[0] + shift_time, x[-1] + shift_time])
        
    return fig, ax


def plot_spectrogram(
    y: npt.ArrayLike, 
    sr: int, 
    shift_time: float = 0.0, 
    shift_array: npt.ArrayLike = np.array([], dtype=np.float32),
    use_plotly=False,
    use_pitch_names=False,
    ax = None,
    show_colorbar=True,
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """
    Plots a Spectrogram graph.

    Parameters
    ----------
    y : array-like
        The waveform data.
    sr : int
        The sample rate of the waveform data.
    shift_time : float, optional
        A time shift to apply to the spectrogram, in seconds (default 0.0).
    use_plotly : bool, optional
        Whether to use Plotly to plot the spectrogram (default False).
    use_pitch_names : bool, optional
        Whether to use pitch names (e.g. C4, D4, etc.) to label the y-axis (default False).

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objs._figure.Figure
        The generated figure object.
    ax : matplotlib.axes.Axes or None
        The generated axes object. If `use_plotly` is True, this value will be None.
    """
    if use_plotly:
        fig = go.Figure()
        Y = librosa.stft(y)
        frequencies = librosa.fft_frequencies(sr=sr)
        times = librosa.times_like(Y)
        fig.add_trace(
            go.Heatmap(
                z=librosa.amplitude_to_db(
                    np.abs(Y), ref=np.max
                ),
                x=times + shift_time,
                y=frequencies,
                colorscale="Viridis",
            )
        )
        ax = None
        fig.update_layout(
            title="Spectrogram",
            xaxis_title="Time(s)",
            yaxis_title="Frequency(Hz)",
            yaxis=dict(range=[0, 10000]),
        )
        if use_pitch_names: # 使用音階名稱顯示y軸
            notes = librosa.hz_to_note(frequencies[1:]) # 建立音階
            fig.update_yaxes(ticktext=notes, tickvals=frequencies[1:])
            fig.update_yaxes(showticklabels=False) # y軸太密集，不顯示ticks
    else:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(y)), ref=np.max
        )
        img = librosa.display.specshow(
            D, x_axis="time", y_axis="log", sr=sr, ax=ax
        )
        if show_colorbar:
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title("Spectrogram")
        ax.set_xlabel("Time(s)")
        ax.set_ylabel("Frequency(Hz)")
        if shift_array.size > 0:
            ax.set_xticks(shift_array - shift_array[0],
                         shift_array)
            ax.autoscale()
        if use_pitch_names: # 使用音階名稱顯示y軸
            y_ticks = ax.get_yticks()[1:]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(librosa.core.hz_to_note(y_ticks))
    return fig, ax

def signal_RMS_analysis(
    y: npt.ArrayLike, 
    shift_time: float = 0.0,
    use_plotly=False
) -> typing.Tuple[plt.Figure, plt.Axes, npt.ArrayLike, npt.ArrayLike]:
    """
    Computes the Root Mean Square (RMS) of a given audio signal and plots the result.

    Parameters
    ----------
    y : npt.ArrayLike
        The audio signal as a 1-dimensional NumPy array or array-like object.
    shift_time : float, optional
        Time shift in seconds to apply to the plot (default 0.0).
    use_plotly : bool, optional
        Whether to use Plotly for plotting (True) or Matplotlib (False) (default False).

    Returns
    -------
    Tuple[plt.Figure, Union[plt.Axes, None], npt.ArrayLike, npt.ArrayLike]
        A tuple containing:
            - fig : plt.Figure or go.Figure
                The plot figure object (either a Matplotlib or Plotly figure object).
            - ax : plt.Axes or None
                The plot axes object (only for Matplotlib) or None if `use_plotly` is True.
            - times : npt.ArrayLike
                A 1-dimensional NumPy array of the times (in seconds) at which the RMS was computed.
            - rms : npt.ArrayLike
                A 1-dimensional NumPy array containing the RMS values for each window.

    Raises
    ------
    TypeError
        If the input signal is not a 1-dimensional NumPy array or array-like object.

    Examples
    --------
    # Compute the RMS and plot the result using Matplotlib
    y, sr = librosa.load('audio_file.wav')
    fig, ax, times, rms = signal_RMS_analysis(y, use_plotly=False)
    plt.show()
    """
    rms = librosa.feature.rms(y = y)
    times = librosa.times_like(rms) + shift_time
    
    if use_plotly:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=rms[0]))
        ax = None
    else:
        fig, ax = plt.subplots()
        ax.plot(times, rms[0])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('RMS')


    return fig, ax, times, rms