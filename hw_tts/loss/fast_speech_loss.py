import torch.nn as nn
import torch

class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_loss = nn.MSELoss()
        self.duration_loss = nn.MSELoss()

    def forward(self, mel, log_duration_predicted, mel_target, duration_predictor_target):
        mel_loss = self.mel_loss(mel, mel_target)

        duration_predictor_loss = self.duration_loss(log_duration_predicted,
                                                     torch.log(duration_predictor_target.float() + 1))

        return mel_loss, duration_predictor_loss
