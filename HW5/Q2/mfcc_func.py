import os
import numpy as np
from scipy.io import wavfile
import cmsisdsp as dsp
import cmsisdsp.mfcc as mfcc
from cmsisdsp.datatype import F32


def create_mfcc_features(recordings_list, FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs, window):
    freq_min = 20
    freq_high = sample_rate / 2

    # Mel filtreleri ve DCT matrisi
    filtLen, filtPos, packedFilters = mfcc.melFilterMatrix(
        F32, freq_min, freq_high, numOfMelFilters, sample_rate, FFTSize
    )
    dctMatrixFilters = mfcc.dctMatrix(F32, numOfDctOutputs, numOfMelFilters)

    num_samples = len(recordings_list)
    mfcc_features = np.empty((num_samples, numOfDctOutputs * 2), np.float32)
    labels = np.empty(num_samples, dtype=np.int32)

    mfccf32 = dsp.arm_mfcc_instance_f32()

    status = dsp.arm_mfcc_init_f32(
        mfccf32,
        FFTSize,
        numOfMelFilters,
        numOfDctOutputs,
        dctMatrixFilters,
        filtPos,
        filtLen,
        packedFilters,
        window,
    )

    for sample_idx, (rec_dir, rec_name) in enumerate(recordings_list):
        wav_path = os.path.join(rec_dir, rec_name)

        wav_file = os.path.basename(wav_path)
        file_specs = os.path.splitext(wav_file)[0]
        digit, person, recording = file_specs.split("_")

        sr, sample = wavfile.read(wav_path)

        if sample.ndim > 1:
            sample = sample[:, 0]

        sample = sample.astype(np.float32)
        sample = sample / np.max(np.abs(sample))

        if len(sample) < 2 * FFTSize:
            padded = np.zeros(2 * FFTSize, dtype=np.float32)
            padded[:len(sample)] = sample
            sample = padded

        first_half = sample[:FFTSize]
        second_half = sample[FFTSize:2 * FFTSize]

        tmp = np.zeros(FFTSize + 2, dtype=np.float32)

        first_half_mfcc = dsp.arm_mfcc_f32(mfccf32, first_half, tmp)
        second_half_mfcc = dsp.arm_mfcc_f32(mfccf32, second_half, tmp)

        mfcc_feature = np.concatenate((first_half_mfcc, second_half_mfcc))

        mfcc_features[sample_idx, :] = mfcc_feature
        labels[sample_idx] = int(digit)

    return mfcc_features, labels