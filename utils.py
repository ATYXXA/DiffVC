# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def save_plot(tensor, savepath):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()

import librosa

def normalize_audio(y):
    rms = librosa.feature.rms(y=y)[0]
    target_rms = 0.1  # 你可以根据需要调整目标响度
    gain = target_rms / np.mean(rms)
    y_normalized = y * gain
    return y_normalized


def save_audio(file_path, sampling_rate, audio, normalize=False):
    audio = np.clip(audio.detach().cpu().squeeze().numpy(), -0.999, 0.999)
    if normalize:
        audio = librosa.util.normalize(audio) * 0.95
        # audio = librosa.effects.preemphasis(audio)
        # audio = normalize_audio(audio)
    wavfile.write(file_path, sampling_rate, (audio * 32767).astype("int16"))
