import os
import glob
import random
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as nnF

# Set this yourself depending where you put the files
dataroot = "."
SAMPLE_RATE = 8000
N_MFCC = 13

INSTRUMENT_MAP = {'guitar': 0, 'vocal': 1}
NUM_CLASSES = len(INSTRUMENT_MAP)

audio_paths = glob.glob(dataroot + "/nsynth_subset/*.wav")
random.seed(0)
random.shuffle(audio_paths)

torch.use_deterministic_algorithms(True)


# 1. Paths, labels, waveforms
def extract_waveform(path):
    waveform, _ = librosa.load(path, sr=SAMPLE_RATE)
    return waveform


def extract_label(path):
    fname = os.path.basename(path)
    instrument = fname.split("_")[0]
    return INSTRUMENT_MAP[instrument]


waveforms = [extract_waveform(p) for p in audio_paths]
labels = [extract_label(p) for p in audio_paths]


class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(2 * N_MFCC, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(nnF.relu(self.bn1(self.conv1(x))))
        x = self.pool2(nnF.relu(self.bn2(self.conv2(x))))
        x = self.pool3(nnF.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 2. MFCC
def extract_mfcc(w):
    mfcc = librosa.feature.mfcc(y=w, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    features = np.concatenate([mfcc_mean, mfcc_std])
    return torch.FloatTensor(features)


# 3. Spectrogram
def extract_spec(w):
    stft = librosa.stft(w)
    spec = np.abs(stft) ** 2
    return torch.FloatTensor(spec)


# 4. Mel-spectrogram
def extract_mel(w, n_mels=128, hop_length=512):
    mel_spec = librosa.feature.melspectrogram(
        y=w,
        sr=SAMPLE_RATE,
        n_mels=n_mels,
        hop_length=hop_length,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (
        mel_spec_db.max() - mel_spec_db.min() + 1e-8
    )
    return torch.FloatTensor(mel_spec_db)


# 5. Constant-Q transform
def extract_q(w):
    w_16k = librosa.resample(w, orig_sr=SAMPLE_RATE, target_sr=16000)
    result = np.abs(librosa.cqt(w_16k, sr=16000))
    return torch.FloatTensor(result)


# 6. Pitch shift
def pitch_shift(w, n):
    y_shift = librosa.effects.pitch_shift(w, sr=SAMPLE_RATE, n_steps=n)
    y_shift = librosa.util.fix_length(y_shift, size=len(w))
    return y_shift


augmented_waveforms = []
augmented_labels = []

for w, y in zip(waveforms, labels):
    augmented_waveforms.append(w)
    augmented_waveforms.append(pitch_shift(w, 1))
    augmented_waveforms.append(pitch_shift(w, -1))
    augmented_labels += [y, y, y]


# 7. Four classes
INSTRUMENT_MAP_7 = {
    'guitar_acoustic': 0,
    'guitar_electronic': 1,
    'vocal_acoustic': 2,
    'vocal_synthetic': 3,
}
NUM_CLASSES_7 = 4


def extract_label_7(path):
    fname = os.path.basename(path)
    key = "_".join(fname.split("_")[:2])
    return INSTRUMENT_MAP_7[key]


# Use mel-spectrograms for the 4-class model
def feature_func_7(w):
    mel_spec = librosa.feature.melspectrogram(
        y=w,
        sr=SAMPLE_RATE,
        n_mels=128,
        hop_length=256,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    return torch.FloatTensor(mel_spec_db)


labels_7 = [extract_label_7(p) for p in audio_paths]

class MLPClassifier_4classes(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.classifier(x)
        return x


model_7 = MLPClassifier_4classes()