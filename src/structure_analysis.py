import numpy as np
import os, sys, librosa
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import IPython.display as ipd
import pandas as pd
from numba import jit

import libfmp.b
import libfmp.c2
import libfmp.c3
import libfmp.c4
import libfmp.c6
import libfmp
from libfmp.b import FloatingBox

from numpy import typing as npt
import typing

@jit(nopython=True)
def compute_sm_dot(X, Y):
    """Computes similarty matrix from feature sequences using dot (inner) product

    Notebook: C4/C4S2_SSM.ipynb

    Args:
        X (np.ndarray): First sequence
        Y (np.ndarray): Second Sequence

    Returns:
        S (float): Dot product
    """
    S = np.dot(np.transpose(X), Y)
    return S

def plot_feature_ssm(X, Fs_X, S, Fs_S, ann, duration, color_ann=None,
                     title='', label='Time (seconds)', time=True,
                     figsize=(5, 6), fontsize=10, clim_X=None, clim=None):
    """Plot SSM along with feature representation and annotations (standard setting is time in seconds)

    Notebook: C4/C4S2_SSM.ipynb

    Args:
        X: Feature representation
        Fs_X: Feature rate of ``X``
        S: Similarity matrix (SM)
        Fs_S: Feature rate of ``S``
        ann: Annotaions
        duration: Duration
        color_ann: Color annotations (see :func:`libfmp.b.b_plot.plot_segments`) (Default value = None)
        title: Figure title (Default value = '')
        label: Label for time axes (Default value = 'Time (seconds)')
        time: Display time axis ticks or not (Default value = True)
        figsize: Figure size (Default value = (5, 6))
        fontsize: Font size (Default value = 10)
        clim_X: Color limits for matrix X (Default value = None)
        clim: Color limits for matrix ``S`` (Default value = None)

    Returns:
        fig: Handle for figure
        ax: Handle for axes
    """
    cmap = libfmp.b.compressed_gray_cmap(alpha=-10)
    fig, ax = plt.subplots(3, 3, gridspec_kw={'width_ratios': [0.1, 1, 0.05],
                                              'wspace': 0.2,
                                              'height_ratios': [0.3, 1, 0.1]},
                           figsize=figsize)
    libfmp.b.plot_matrix(X, Fs=Fs_X, ax=[ax[0, 1], ax[0, 2]], clim=clim_X,
                         xlabel='', ylabel='', title=title)
    ax[0, 0].axis('off')
    libfmp.b.plot_matrix(S, Fs=Fs_S, ax=[ax[1, 1], ax[1, 2]], cmap=cmap, clim=clim,
                         title='', xlabel='', ylabel='', colorbar=True)
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    libfmp.b.plot_segments(ann, ax=ax[2, 1], time_axis=time, fontsize=fontsize,
                           colors=color_ann,
                           time_label=label, time_max=duration*Fs_X)
    ax[2, 2].axis('off')
    ax[2, 0].axis('off')
    libfmp.b.plot_segments(ann, ax=ax[1, 0], time_axis=time, fontsize=fontsize,
                           direction='vertical', colors=color_ann,
                           time_label=label, time_max=duration*Fs_X)
    return fig, ax

def SSM_chorma(wav_fn:str, anno_fn: str, hop_size: int = 4096, Nfft: int = 1024) -> None :
    
    x, fs = librosa.load(wav_fn)
    duration= (x.shape[0])/fs

    chromagram = librosa.feature.chroma_stft(y=x, sr=fs, tuning=0, norm=2, hop_length=hop_size, n_fft=Nfft)
    X, Fs_X = libfmp.c3.smooth_downsample_feature_sequence(chromagram, fs/hop_size, filt_len=41, down_sampling=10)

    # According to the documentation
    ann, color_ann = libfmp.c4.read_structure_annotation(os.path.join(anno_fn), fn_ann_color=anno_fn)
    ann_frames = libfmp.c4.convert_structure_annotation(ann, Fs=Fs_X) 
    
    X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
    S = compute_sm_dot(X,X)
    fig, ax = plot_feature_ssm(X, 1, S, 1, ann_frames, duration*Fs_X, color_ann=color_ann,
                               clim_X=[0,1], clim=[0,1], label='Time (frames)',
                               title='Chroma feature (Fs=%0.2f)'%Fs_X)
    return fig, ax

def plot_self_similarity(y_ref: npt.ArrayLike, sr: int, affinity: bool = False, hop_length: int = 1024) -> None:
    '''
    To visualize the similarity matrix of the signal

    y_ref: reference signal
    y_comp: signal to be compared
    sr: sampling rate
    affinity: to use affinity or not
    hop_size
    '''


    # Pre-processing stage
    chroma = librosa.feature.chroma_cqt(y=y_ref, sr=sr, hop_length=hop_length)
    chroma_stack = librosa.feature.stack_memory(chroma, n_steps=10, delay=3)

    fig, ax = plt.subplots()

    if not affinity:
        R = librosa.segment.recurrence_matrix(chroma_stack, k=5)
        imgsim = librosa.display.specshow(R, x_axis='s', y_axis='s',
                                        hop_length=hop_length)
        plt.title('Binary recurrence (symmetric)')
        plt.colorbar()

    else:
        R_aff = librosa.segment.recurrence_matrix(chroma_stack, metric='cosine',mode='affinity')
        imgaff = librosa.display.specshow(R_aff, x_axis='s', y_axis='s',
                                        cmap='magma_r', hop_length=hop_length)
        plt.title('Affinity recurrence')
        plt.colorbar()

    return fig, ax

@jit(nopython=True)
def compute_kernel_checkerboard_gaussian(L: int =10 , var: float = 0.5, normalize=True) -> npt.ArrayLike:
    """Compute Guassian-like checkerboard kernel [FMP, Section 4.4.1].
    See also: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        L (int): Parameter specifying the kernel size M=2*L+1
        var (float): Variance parameter determing the tapering (epsilon) (Default value = 1.0)
        normalize (bool): Normalize kernel (Default value = True)

    Returns:
        kernel (np.ndarray): Kernel matrix of size M x M
    """
    taper = np.sqrt(1/2) / (L * var)
    axis = np.arange(-L, L+1)
    gaussian1D = np.exp(-taper**2 * (axis**2))
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    kernel_box = np.outer(np.sign(axis), np.sign(axis))
    kernel = kernel_box * gaussian2D
    if normalize:
        kernel = kernel / np.sum(np.abs(kernel))
    return kernel


def compute_novelty_ssm(S, kernel: npt.ArrayLike = None, L: int = 10, var: float = 0.5, exclude: bool =False) -> npt.ArrayLike:
    """Compute novelty function from SSM [FMP, Section 4.4.1]

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        S (np.ndarray): SSM
        kernel (np.ndarray): Checkerboard kernel (if kernel==None, it will be computed) (Default value = None)
        L (int): Parameter specifying the kernel size M=2*L+1 (Default value = 10)
        var (float): Variance parameter determing the tapering (epsilon) (Default value = 0.5)
        exclude (bool): Sets the first L and last L values of novelty function to zero (Default value = False)

    Returns:
        nov (np.ndarray): Novelty function
    """
    if kernel is None:
        kernel = compute_kernel_checkerboard_gaussian(L=L, var=var)
    N = S.shape[0]
    M = 2*L + 1
    nov = np.zeros(N)
    # np.pad does not work with numba/jit
    S_padded = np.pad(S, L, mode='constant')

    for n in range(N):
        # Does not work with numba/jit
        nov[n] = np.sum(S_padded[n:n+M, n:n+M] * kernel)
    if exclude:
        right = np.min([L, N])
        left = np.max([0, N-L])
        nov[0:right] = 0
        nov[left:N] = 0

    return nov

def SSM_Novelty(wav_filename:str, anno_csv_filename: str) -> None :

    float_box = libfmp.b.FloatingBox()

    fn_wav = os.path.join(wav_filename)
    ann, color_ann = libfmp.c4.read_structure_annotation(os.path.join(anno_csv_filename), 
                                                         fn_ann_color=anno_csv_filename)

    S_dict = {}
    Fs_dict = {}
    x, x_duration, X, Fs_X, S, I = libfmp.c4.compute_sm_from_filename(fn_wav, 
                                                    L=11, H=5, L_smooth=1, thresh=1)

    S_dict[0], Fs_dict[0] = S, Fs_X
    ann_frames = libfmp.c4.convert_structure_annotation(ann, Fs=Fs_X) 
    fig, ax = libfmp.c4.plot_feature_ssm(X, 1, S, 1, ann_frames, x_duration*Fs_X,
                label='Time (frames)', color_ann=color_ann, clim_X=[0,1], clim=[0,1], 
                title='Feature rate: %0.0f Hz'%(Fs_X), figsize=(4.5, 5.5))
    float_box.add_fig(fig)

    x, x_duration, X, Fs_X, S, I = libfmp.c4.compute_sm_from_filename(fn_wav, 
                                                    L=41, H=10, L_smooth=1, thresh=1)
    S_dict[1], Fs_dict[1] = S, Fs_X
    ann_frames = libfmp.c4.convert_structure_annotation(ann, Fs=Fs_X) 
    fig, ax = libfmp.c4.plot_feature_ssm(X, 1, S, 1, ann_frames, x_duration*Fs_X,
                label='Time (frames)', color_ann=color_ann, clim_X=[0,1], clim=[0,1], 
                title='Feature rate: %0.0f Hz'%(Fs_X), figsize=(4.5, 5.5))
    float_box.add_fig(fig)
    float_box.show()

    figsize=(10,6)
    L_kernel_set = [5, 10, 20, 40]
    num_kernel = len(L_kernel_set)
    num_SSM = len(S_dict)

    fig, ax = plt.subplots(num_kernel, num_SSM, figsize=figsize)
    for s in range(num_SSM):
        for t in range(num_kernel):
            L_kernel = L_kernel_set[t]
            S = S_dict[s]
            nov = compute_novelty_ssm(S, L=L_kernel, exclude=True)        
            fig_nov, ax_nov, line_nov = libfmp.b.plot_signal(nov, Fs = Fs_dict[s], 
                    color='k', ax=ax[t,s], figsize=figsize, 
                    title='Feature rate = %0.0f Hz, $L_\mathrm{kernel}$ = %d'%(Fs_dict[s],L_kernel)) 
            libfmp.b.plot_segments_overlay(ann, ax=ax_nov, colors=color_ann, alpha=0.1, 
                                           edgecolor='k', print_labels=False)
    plt.tight_layout()
    plt.show()  
    