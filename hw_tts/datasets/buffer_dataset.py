import time
import torch
import os
import numpy as np

from torch.utils.data import Dataset
from tqdm import tqdm

from utils import process_text
from hw_tts.text import text_to_sequence


def get_data_to_buffer(data_path, mel_ground_truth, alignment_path, text_cleaners):
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

        buffer.append({"text": character, "duration": duration,
                       "mel_target": mel_gt_target})

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end - start))

    return buffer


class BufferDataset(Dataset):
    def __init__(self, data_path, mel_ground_truth, alignment_path, text_cleaners):
        self.buffer = get_data_to_buffer(data_path, mel_ground_truth, alignment_path, text_cleaners)
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]
