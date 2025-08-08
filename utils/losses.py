import torch.nn.functional as F

def sse_loss(pred, target):
    return F.mse_loss(pred, target) #sum over samples (x-x_hat)^2

import torch
import torchaudio
import torch.nn as nn

class MelLoss(nn.Module):
    def __init__(self, sample_rate=44100, n_fft=512, hop_length=480, n_mels=40):
        super(MelLoss, self).__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,  # power=2.0 means computing power spectrogram
            center=False
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=None)

    def forward(self, pred_waveform, target_waveform):
        if pred_waveform.ndim == 3:
            pred_waveform = pred_waveform.squeeze(1)
        if target_waveform.ndim == 3:
            target_waveform = target_waveform.squeeze(1)
        
        mel_pred = self.db_transform(self.mel_spec(pred_waveform))
        mel_target = self.db_transform(self.mel_spec(target_waveform))
        return torch.mean((mel_pred - mel_target) ** 2)


import numpy as np

class PriorityWeightingLossPAM1(nn.Module):
    def __init__(self, sample_rate=44100, n_fft=512, hop_length=480):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft)

        freqs = np.fft.rfftfreq(n_fft, d=1.0/sample_rate)
        self.bark = torch.tensor([13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2) for f in freqs], dtype=torch.float32)
        self.ath_power = torch.tensor(self._absolute_threshold(freqs), dtype=torch.float32)

    def _absolute_threshold(self, f):
        f = np.maximum(f, 1e-6)
        ath_db = 3.64*(f/1000)**(-0.8) - 6.5*np.exp(-0.6*(f/1000 - 3.3)**2) + 1e-3*(f/1000)**4
        ath_db = np.clip(ath_db, -100, 100)
        ath_power = np.maximum(10 ** (ath_db / 10), 1e-12)
        return ath_power
    def _spreading_function(self, df):
        spread_db = torch.where(df >= 0, -17 * df, -27 * df.abs())
        return 10 ** (spread_db / 10)
    def forward(self, pred, target):
        if pred.ndim == 3:
            pred = pred.squeeze(1)
        if target.ndim == 3:
            target = target.squeeze(1)

        
        S_pred = torch.stft(pred, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True, window=self.window.to(pred.device))
        S_target = torch.stft(target, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True, window=self.window.to(target.device))

        power_pred = S_pred.abs().pow(2) + 1e-12
        power_target = S_target.abs().pow(2) + 1e-12

        
        psd_db = 10 * power_target.log10()
        B, F, T = psd_db.shape

        masking_power = torch.zeros_like(power_target)

       
        for b in range(B):
            for t in range(T):
                spectrum = psd_db[b, :, t]
                tonal_indices = (spectrum[1:-1] > spectrum[:-2]) & (spectrum[1:-1] > spectrum[2:])
                tonal_indices = torch.nonzero(tonal_indices).squeeze() + 1  # shift index due to slicing

                for idx in tonal_indices:
                    if spectrum[idx] < -40:
                        continue
                    masker_bark = self.bark[idx]
                    masker_power = 10 ** (spectrum[idx].item() / 10)

                    df = self.bark - masker_bark
                    spread = self._spreading_function(df.to(pred.device))
                    spread_power = masker_power * spread

                    masking_power[b, :, t] = torch.maximum(masking_power[b, :, t], spread_power)

        combined_power = masking_power + self.ath_power.to(pred.device)[None, :, None]
        m_f = combined_power  # [B, F, T]
        p_f = power_target    # [B, F, T]

        
        weights = torch.log10((p_f / (m_f + 1e-12)) + 1.0)

        
        mag_pred = torch.sqrt(power_pred + 1e-12)
        mag_target = torch.sqrt(p_f + 1e-12)
        freq_error = (mag_pred - mag_target).pow(2)
        weighted_error = weights * freq_error

        return weighted_error.mean()
