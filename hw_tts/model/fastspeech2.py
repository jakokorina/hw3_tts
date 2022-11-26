import torch
import torch.nn as nn

from hw_tts.encoder_decoder import Encoder, Decoder
from hw_tts.length_regulator import LengthRegulator


def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self, max_seq_len, encoder_n_layer, decoder_n_layer, vocab_size, encoder_dim, PAD,
                 encoder_conv1d_filter_size, encoder_head, fft_conv1d_kernel,
                 fft_conv1d_padding, duration_predictor_filter_size,
                 duration_predictor_kernel_size, num_mels, dropout=0.1, **kwargs):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(max_seq_len, encoder_n_layer, vocab_size, encoder_dim, PAD,
                               encoder_conv1d_filter_size, encoder_head, fft_conv1d_kernel,
                               fft_conv1d_padding, dropout)
        self.length_regulator = LengthRegulator(encoder_dim, duration_predictor_filter_size,
                                                duration_predictor_kernel_size, dropout)
        self.decoder = Decoder(max_seq_len, decoder_n_layer, encoder_dim, PAD,
                               encoder_conv1d_filter_size, encoder_head, fft_conv1d_kernel,
                               fft_conv1d_padding, dropout)

        self.mel_linear = nn.Linear(encoder_dim, num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0):
        x, mask = self.encoder(src_seq, src_pos)

        if self.training:
            output, duration_preditor_output = self.length_regulator(x, alpha, length_target, mel_max_length)
            output = self.decoder(output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
            return output, duration_preditor_output

        else:
            output, mel_pos = self.length_regulator(x, alpha)
            output = self.decoder(output, mel_pos)
            output = self.mel_linear(output)
            return output
