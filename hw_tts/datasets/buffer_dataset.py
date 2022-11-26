import time
import torch
import os
import librosa
import numpy as np
import pyworld as pw

from torch.utils.data import Dataset
from tqdm import tqdm

from hw_tts.waveglow.utils import process_text
from hw_tts.text import text_to_sequence
import hw_tts.audio.hparams_audio as hparams_audio


def get_data_to_buffer(data_path, mel_ground_truth, alignment_path, pitch_paths,
                       energy_paths, text_cleaners):
    buffer = list()
    text = process_text(data_path)
    start = time.perf_counter()
    for i in tqdm(range(len(text))):
        mel_gt_name = os.path.join(
            mel_ground_truth, "ljspeech-mel-%05d.npy" % (i + 1))
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(os.path.join(
            alignment_path, str(i) + ".npy"))
        character = text[i][0:len(text[i]) - 1]
        character = np.array(
            text_to_sequence(character, text_cleaners))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)

        pitch = torch.load(os.path.join(pitch_paths, f"pitch_{i}.pt"))
        energy = torch.load(os.path.join(energy_paths, f"{i}.pt"))

        buffer.append({"text": character, "duration": duration, "mel_target": mel_gt_target,
                       "pitch": pitch, "energy": energy})

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end - start))

    return buffer


class BufferDataset(Dataset):
    def __init__(self, data_path, mel_ground_truth, alignment_path, pitch_paths, energy_paths, text_cleaners):
        self.buffer = get_data_to_buffer(data_path, mel_ground_truth, alignment_path, pitch_paths, energy_paths,
                                         text_cleaners)
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]
