import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import librosa
import libfmp.b
import libfmp.c3
import libfmp.c4

import sys

def compute_chromagram_from_filename(fn_wav, Fs=22050, N=4096, H=2048, gamma=None, version='STFT', norm='2'):
    """Compute chromagram for WAV file specified by filename

    Notebook: C5/C5S2_ChordRec_Templates.ipynb

    Args:
        fn_wav (str): Filenname of WAV
        Fs (scalar): Sampling rate (Default value = 22050)
        N (int): Window size (Default value = 4096)
        H (int): Hop size (Default value = 2048)
        gamma (float): Constant for logarithmic compression (Default value = None)
        version (str): Technique used for front-end decomposition ('STFT', 'IIS', 'CQT') (Default value = 'STFT')
        norm (str): If not 'None', chroma vectors are normalized by norm as specified ('1', '2', 'max')
            (Default value = '2')

    Returns:
        X (np.ndarray): Chromagram
        Fs_X (scalar): Feature reate of chromagram
        x (np.ndarray): Audio signal
        Fs (scalar): Sampling rate of audio signal
        x_dur (float): Duration (seconds) of audio signal
    """
    x, Fs = librosa.load(fn_wav, sr=Fs)
    x_dur = x.shape[0] / Fs
    if version == 'STFT':
        # Compute chroma features with STFT
        X = librosa.stft(x, n_fft=N, hop_length=H, pad_mode='constant', center=True)
        if gamma is not None:
            X = np.log(1 + gamma * np.abs(X) ** 2)
        else:
            X = np.abs(X) ** 2
        X = librosa.feature.chroma_stft(S=X, sr=Fs, tuning=0, norm=None, hop_length=H, n_fft=N)
    if version == 'CQT':
        # Compute chroma features with CQT decomposition
        X = librosa.feature.chroma_cqt(y=x, sr=Fs, hop_length=H, norm=None)
    if version == 'IIR':
        # Compute chroma features with filter bank (using IIR elliptic filter)
        X = librosa.iirt(y=x, sr=Fs, win_length=N, hop_length=H, center=True, tuning=0.0)
        if gamma is not None:
            X = np.log(1.0 + gamma * X)
        X = librosa.feature.chroma_cqt(C=X, bins_per_octave=12, n_octaves=7,
                                       fmin=librosa.midi_to_hz(24), norm=None)
    if norm is not None:
        X = libfmp.c3.normalize_feature_sequence(X, norm=norm)
    Fs_X = Fs / H
    return X, Fs_X, x, Fs, x_dur

def compute_chromagram(y, sr, Fs=22050, N=4096, H=2048, gamma=None, version='STFT', norm='2'):
    """Compute chromagram for WAV file specified by filename

    Notebook: C5/C5S2_ChordRec_Templates.ipynb

    Args:
        y (np.ndarray): Audio signal
        sr (scalar): Sampling rate
        Fs (scalar): Sampling rate (Default value = 22050)
        N (int): Window size (Default value = 4096)
        H (int): Hop size (Default value = 2048)
        gamma (float): Constant for logarithmic compression (Default value = None)
        version (str): Technique used for front-end decomposition ('STFT', 'IIS', 'CQT') (Default value = 'STFT')
        norm (str): If not 'None', chroma vectors are normalized by norm as specified ('1', '2', 'max')
            (Default value = '2')

    Returns:
        X (np.ndarray): Chromagram
        Fs_X (scalar): Feature reate of chromagram
        x (np.ndarray): Audio signal
        Fs (scalar): Sampling rate of audio signal
        x_dur (float): Duration (seconds) of audio signal
    """
    x = librosa.resample(y, sr, Fs)
    x_dur = x.shape[0] / Fs
    if version == 'STFT':
        # Compute chroma features with STFT
        X = librosa.stft(x, n_fft=N, hop_length=H, pad_mode='constant', center=True)
        if gamma is not None:
            X = np.log(1 + gamma * np.abs(X) ** 2)
        else:
            X = np.abs(X) ** 2
        X = librosa.feature.chroma_stft(S=X, sr=Fs, tuning=0, norm=None, hop_length=H, n_fft=N)
    if version == 'CQT':
        # Compute chroma features with CQT decomposition
        X = librosa.feature.chroma_cqt(y=x, sr=Fs, hop_length=H, norm=None)
    if version == 'IIR':
        # Compute chroma features with filter bank (using IIR elliptic filter)
        X = librosa.iirt(y=x, sr=Fs, win_length=N, hop_length=H, center=True, tuning=0.0)
        if gamma is not None:
            X = np.log(1.0 + gamma * X)
        X = librosa.feature.chroma_cqt(C=X, bins_per_octave=12, n_octaves=7,
                                       fmin=librosa.midi_to_hz(24), norm=None)
    if norm is not None:
        X = libfmp.c3.normalize_feature_sequence(X, norm=norm)
    Fs_X = Fs / H
    return X, Fs_X, x, Fs, x_dur

def get_chord_labels(ext_minor='m', nonchord=False):
    """Generate chord labels for major and minor triads (and possibly nonchord label)

    Notebook: C5/C5S2_ChordRec_Templates.ipynb

    Args:
        ext_minor (str): Extension for minor chords (Default value = 'm')
        nonchord (bool): If "True" then add nonchord label (Default value = False)

    Returns:
        chord_labels (list): List of chord labels
    """
    chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chord_labels_maj = chroma_labels
    chord_labels_min = [s + ext_minor for s in chroma_labels]
    chord_labels = chord_labels_maj + chord_labels_min
    if nonchord is True:
        chord_labels = chord_labels + ['N']
    return chord_labels

def generate_chord_templates(nonchord=False):
    """Generate chord templates of major and minor triads (and possibly nonchord)

    Notebook: C5/C5S2_ChordRec_Templates.ipynb

    Args:
        nonchord (bool): If "True" then add nonchord template (Default value = False)

    Returns:
        chord_templates (np.ndarray): Matrix containing chord_templates as columns
    """
    template_cmaj = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]).T
    template_cmin = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]).T
    num_chord = 24
    if nonchord:
        num_chord = 25
    chord_templates = np.ones((12, num_chord))
    for shift in range(12):
        chord_templates[:, shift] = np.roll(template_cmaj, shift)
        chord_templates[:, shift+12] = np.roll(template_cmin, shift)
    return chord_templates

def chord_recognition_template(X, norm_sim='1', nonchord=False):
    """Conducts template-based chord recognition
    with major and minor triads (and possibly nonchord)

    Notebook: C5/C5S2_ChordRec_Templates.ipynb

    Args:
        X (np.ndarray): Chromagram
        norm_sim (str): Specifies norm used for normalizing chord similarity matrix (Default value = '1')
        nonchord (bool): If "True" then add nonchord template (Default value = False)

    Returns:
        chord_sim (np.ndarray): Chord similarity matrix
        chord_max (np.ndarray): Binarized chord similarity matrix only containing maximizing chord
    """
    chord_templates = generate_chord_templates(nonchord=nonchord)
    X_norm = libfmp.c3.normalize_feature_sequence(X, norm='2')
    chord_templates_norm = libfmp.c3.normalize_feature_sequence(chord_templates, norm='2')
    chord_sim = np.matmul(chord_templates_norm.T, X_norm)
    if norm_sim is not None:
        chord_sim = libfmp.c3.normalize_feature_sequence(chord_sim, norm=norm_sim)
    # chord_max = (chord_sim == chord_sim.max(axis=0)).astype(int)
    chord_max_index = np.argmax(chord_sim, axis=0)
    chord_max = np.zeros(chord_sim.shape).astype(np.int32)
    for n in range(chord_sim.shape[1]):
        chord_max[chord_max_index[n], n] = 1

    return chord_sim, chord_max

def plot_chord_recognition(y, sr) :
    import warnings
    warnings.warn("This function is deprecated and will be removed in future versions.", DeprecationWarning)
    
    X, Fs_X, x, Fs, x_dur = compute_chromagram(y, sr)
    
    chord_sim, chord_max = chord_recognition_template(X, norm_sim='max')
    chord_labels = get_chord_labels(nonchord=False)

    cmap = libfmp.b.compressed_gray_cmap(alpha=1, reverse=False)
    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.03], 
                                          'height_ratios': [1.5, 3]}, figsize=(8, 10))

    libfmp.b.plot_chromagram(X, ax=[ax[0,0], ax[0,1]], Fs=Fs_X, clim=[0, 1], xlabel='',
                         title='STFT-based chromagram (feature rate = %0.1f Hz)' % (Fs_X))
    libfmp.b.plot_matrix(chord_max, ax=[ax[1, 0], ax[1, 1]], Fs=Fs_X, 
                     title='Time–chord representation of chord recognition result',
                     ylabel='Chord', xlabel='')
    ax[1, 0].set_yticks(np.arange( len(chord_labels) ))
    ax[1, 0].set_yticklabels(chord_labels)
    ax[1, 0].grid()
    plt.tight_layout()
    return fig, ax, chord_max

def plot_binary_template_chord_recognition(y, sr) :
    import warnings
    warnings.warn("This function is deprecated and will be removed in future versions.", DeprecationWarning)
    
    X, Fs_X, x, Fs, x_dur = compute_chromagram(y, sr)
    chord_sim, chord_max = chord_recognition_template(X, norm_sim='max')

    chord_templates = generate_chord_templates()
    X_chord = np.matmul(chord_templates, chord_max)

    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.03], 
                                              'height_ratios': [1, 1]}, figsize=(8, 5))

    libfmp.b.plot_chromagram(X, ax=[ax[0, 0], ax[0, 1]], Fs=Fs_X, clim=[0, 1], xlabel='',
                            title='STFT-based chromagram (feature rate = %0.1f Hz)' % (Fs_X))
    libfmp.b.plot_chromagram(X_chord, ax=[ax[1, 0], ax[1, 1]], Fs=Fs_X, clim=[0, 1], xlabel='',
                            title='Binary templates of the chord recognition result')
    plt.tight_layout()
    return fig, ax


def chord_table(chord_max):
    
    chord_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] + ['Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm']
    
    # 計算chord_max依照第一個軸的最大值的index
    chord_max_index = np.argmax(chord_max, axis=0)
    # 用index找出對應的chord_labels
    chord_results = [chord_labels[i] for i in chord_max_index]
    
    return chord_results


def plot_chord(chroma, title="", figsize=(12, 6), cmap="coolwarm", include_minor=False):
    import seaborn as sns
    chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if include_minor:
        chroma_labels += ['Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(chroma, ax=ax, cmap=cmap, linewidths=0.01, linecolor=(1, 1, 1, 0.1))
    ax.invert_yaxis()
    ax.set_yticks(
        np.arange(len(chroma_labels)) + 0.5,
        chroma_labels,
        rotation=0,
    )
    ax.set_ylabel("Chord")
    ax.set_xlabel('Time (frame)')
    ax.set_title(title)
    
    return fig, ax

def plot_user_chord(
    df,
    ax = None
):
    
    import seaborn as sns
    chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] + ['Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm']
    
    # 檢查df["Chord"]無chroma_labels以外的值
    assert df["Chord"].isin(chroma_labels).all(), "Chord must be in chroma_labels"
    
    # 將df["Chord"]轉成chroma_labels的index
    chord_index = df["Chord"].apply(lambda x: chroma_labels.index(x))
    
    # 建立一個24 * len(df)的矩陣，並將值設為0
    chroma = np.zeros((24, len(df)))
    # 依照chord_index的值將chroma的值設為1
    chroma[chord_index, np.arange(len(df)),] = 1
    
    # 繪圖
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()
    sns.heatmap(chroma, ax=ax, cmap='crest', linewidths=0.01, linecolor=(1, 1, 1, 0.1), cbar=False)
    ax.invert_yaxis()
    ax.set_yticks(
        np.arange(len(chroma_labels)) + 0.5,
        chroma_labels,
        rotation=0,
    )
    ax.set_ylabel("Chord")
    ax.set_xlabel('Time (frame)')
    ax.set_title('User Chord Recognition Result')
    
    return fig, ax


chord_color_map = {
    "C": "red",
    "D": "green",
    "E": "blue",
    "F": "purple",
    "G": "orange",
    "A": "pink",
    "B": "brown"
}

def plot_chord_block(
    chord_df,
    shift_time=0.0,
    ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 3))
    else:
        fig = ax.get_figure()

    df = chord_df.copy()
    grouped = df.groupby((df['Chord'] != df['Chord'].shift()).cumsum())
    chord_grouped_df = pd.DataFrame(columns=["Start", "End", "Chord"])
    for group_id, group_df in grouped:
        start_index = group_df.index[0]
        end_index = group_df.index[-1]
        chord_value = group_df.iloc[0]["Chord"]
        chord_grouped_df = chord_grouped_df.append({
            "Start": start_index,
            "End": end_index,
            "Chord": chord_value
            },
            ignore_index=True
        )  
        
    # 繪圖
    for index, row in chord_grouped_df.iterrows():
        start = row["Start"]
        end = row["End"] + 1
        chord = row["Chord"]
        color = chord_color_map[chord[0]]
        alpha = 0.8 if len(chord) == 2 else 0.5
        
        ax.axvspan(start, end, alpha=alpha, color=color)
        ax.text((start+end)/2, 0.5, chord, ha='center', va='center', rotation=0, size=10)
    # 不顯示y軸
    ax.axes.yaxis.set_visible(False)
    # 不顯示x軸
    ax.axes.xaxis.set_visible(False)
    # 設定x軸範圍
    ax.set_xlim(0, chord_df.shape[0])
    # 設定標題
    ax.set_title("Chord Recognition Result")
    
        
    return fig, ax