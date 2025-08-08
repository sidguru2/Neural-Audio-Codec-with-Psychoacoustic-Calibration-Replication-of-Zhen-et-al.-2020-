import os
from glob import glob
import torch
import torchaudio
from model.modules import Encoder, Decoder, SoftToHardQuantizer
from utils.audio_io import save_audio
import math
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#model_name = "/Users/sidabid09/nac_model/results/model_aweights.pth"
#just while training
import shutil
shutil.copy("modelc_1ae_weights.pth", "modelb_eval_snapshot6.pth")
checkpoint = torch.load("modelb_eval_snapshot6.pth", map_location=device)

#checkpoint = torch.load("modelb_1ae_weights.pth", map_location=device)
model_type = '1AE'
ldim = 8 #8 for 2ae
if model_type == '1AE':
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
elif model_type == '2AE':
    encoder1 = Encoder().to(device)
    decoder1 = Decoder().to(device)
    encoder2 = Encoder().to(device)
    decoder2 = Decoder().to(device)
else:
    raise ValueError("Invalid model_type: must be '1AE' or '2AE'")
quantizer = SoftToHardQuantizer(num_kernels=64, alpha=10.0).to(device)


num_codebook_entries = 64
def make_partial_hann_window(frame_size=512, overlap=32):
    window = torch.ones(frame_size)
    hann_overlap = torch.hann_window(overlap * 2)

    window[:overlap] = hann_overlap[:overlap]  # fade-in
    window[-overlap:] = hann_overlap[overlap:]  # fade-out

    return window

partial_hann_window = make_partial_hann_window(frame_size=512, overlap=32)


if model_type == '1AE':
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
elif model_type == '2AE':
    encoder1.load_state_dict(checkpoint['encoder1'])
    decoder1.load_state_dict(checkpoint['decoder1'])
    encoder2.load_state_dict(checkpoint['encoder2'])
    decoder2.load_state_dict(checkpoint['decoder2'])

quantizer.load_state_dict(checkpoint['quantizer'])


#print(f"Number of codebook entries: {num_codebook_entries}")

#def frame_audio(wav, frame_size=512, hop_size=480):
#    return wav.unfold(1, frame_size, hop_size).squeeze(0)
def frame_audio(wav, frame_size=512, hop_size=480, window=None):
    frames = wav.unfold(1, frame_size, hop_size).squeeze(0)  # (num_frames, frame_size)
    if window is not None:
        frames = frames * window  # apply window to each frame
    return frames

data_root = "data/genres_original"
all_wav_paths = glob(os.path.join(data_root, "*", "*.wav"))

all_frames = []

for path in all_wav_paths:
    waveform, sr = torchaudio.load(path)
    if sr != 44100:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=44100)
    waveform = waveform.mean(dim=0, keepdim=True)

    if waveform.shape[1] >= 512:
        frames = frame_audio(waveform, frame_size=512, hop_size=480, window=partial_hann_window) #instead of just waveform
        if frames.shape[1] == 512:
            for frame in frames:
                all_frames.append((frame.unsqueeze(0)))


frames = torch.stack(all_frames[:2000]).to(device)

# inference
with torch.no_grad():
    #z = encoder(frames)
    #z_q, indices = quantizer(z, hard=True)
    
    #recon_frames = decoder(z_q)
    if model_type == '1AE':
        z = encoder(frames)
        z_q, _ = quantizer(z, hard=True)
        recon_frames = decoder(z_q)
    elif model_type == '2AE':
        z1 = encoder1(frames)
        print("z1 std:", z1.std().item())
        z1_q, _ = quantizer(z1, hard=True)
        x1_hat = decoder1(z1_q)

        z2 = encoder2(x1_hat)
        z2_q, _ = quantizer(z2, hard=True)
        recon_frames = decoder2(z2_q)
        import matplotlib.pyplot as plt
        plt.plot(x1_hat[0].cpu().squeeze().numpy())
        plt.title("Decoder1 output")
        plt.show()

        # Plot final reconstruction
        plt.plot(recon_frames[0].cpu().squeeze().numpy())
        plt.title("Decoder2 (final) output")
        plt.show()
    #nique_indices = torch.unique(indices)
    #used_entries = unique_indices.numel() #unnecessary, no need for effective bitrate
    
print("Recon frames shape:", recon_frames.shape)
print("Recon min/max:", recon_frames.min().item(), recon_frames.max().item())

import matplotlib.pyplot as plt
plt.plot(recon_frames[0].cpu().squeeze().numpy())
plt.title("Sample decoded frame")
plt.show()


# from utils - audio.io
#def overlap_add(frames, frame_size=512, hop_size=480):
#    num_frames = frames.shape[0]
#    signal_length = (num_frames - 1) * hop_size + frame_size
#    output = torch.zeros(signal_length)
#
#    for i in range(num_frames):
#        start = i * hop_size
#        output[start:start+frame_size] += frames[i, 0]
#
#    return output / output.abs().max()
#new for hann windowing
def overlap_add(frames, frame_size=512, hop_size=480, window=None):
    num_frames = frames.shape[0]
    signal_length = (num_frames - 1) * hop_size + frame_size

    output = torch.zeros(signal_length)
    window_sum = torch.zeros(signal_length)

    for i in range(num_frames):
        start = i * hop_size
        output[start:start+frame_size] += frames[i, 0] * window
        window_sum[start:start+frame_size] += window

    # Normalize by accumulated window energy to prevent amplitude distortion
    nonzero = window_sum != 0
    output[nonzero] /= window_sum[nonzero]

    # Normalize final waveform
    return output / output.abs().max()

    
#torchaudio.save("single_frame.wav", recon_frames[0].cpu(), 44100)


#reconstructed_signal = overlap_add(recon_frames.cpu()).unsqueeze(0)
##save_audio(reconstructed_signal, "reconstructed_model_a_2ae.wav")
#torchaudio.save("reconstructed_model_a_2ae34.wav", reconstructed_signal, 44100)
#new for hann windowing
import torch

frame_size = 512
hann_window = torch.hann_window(frame_size)
reconstructed_signal = overlap_add(recon_frames.cpu(), frame_size=512, hop_size=480, window=partial_hann_window)
torchaudio.save("reconstructed_modelb1ae_300softmax.wav", reconstructed_signal.unsqueeze(0), 44100)
#reconstructed_decoder1 = overlap_add(x1_hat.cpu(), frame_size=512, hop_size=480, window=partial_hann_window)
#torchaudio.save("reconstructed_modelb2ae_decoder1_again.wav", reconstructed_decoder1.unsqueeze(0), 44100)

#torchaudio.save("original_frame.wav", frames[0].cpu(), 44100)
#torchaudio.save("reconstructed_frame.wav", recon_frames[0].cpu(), 44100)

print("Reconstructed signal max amplitude:", reconstructed_signal.abs().max().item())


def compute_snr(x, x_hat, eps=1e-8):
    #dB
    noise = x - x_hat
    snr = 10 * torch.log10(x.pow(2).mean() / (noise.pow(2).mean() + eps))
    return snr.item()

def compute_lsd(x, x_hat, eps=1e-8):
    #log spectral distance over stft frames
    # assumes x and x_hat are (1, samples)
    X = torch.fft.rfft(x, dim=-1).abs() + eps
    X_hat = torch.fft.rfft(x_hat, dim=-1).abs() + eps
    log_diff = (X.log() - X_hat.log()) ** 2
    lsd = log_diff.mean().sqrt()
    return lsd.item()
    
#reconstructed_signal = overlap_add(recon_frames.cpu())
#original_signal = overlap_add(frames.cpu(), window=None)
window = torch.hann_window(512)
original_signal = overlap_add(frames.cpu(), window=window)


#reconstructed_signal = reconstructed_signal.unsqueeze(0)
original_signal = original_signal.unsqueeze(0)
torchaudio.save("original_signal.wav", original_signal, 44100)


snr = compute_snr(original_signal, reconstructed_signal)
lsd = compute_lsd(original_signal, reconstructed_signal)

#  raw bitrate
bits_per_code = math.log2(num_codebook_entries)
codes_per_second = 44100 / 480
num_code_vectors_per_step = ldim#z_q.shape[1] if z_q.ndim > 2 else 1  # adjust if your quantizer returns multi-vector output
raw_bitrate = codes_per_second * num_code_vectors_per_step * bits_per_code




#unique_indices = torch.unique(indices)
#used_entries = unique_indices.numel()
#used_bits_per_code = math.log2(used_entries)
#efficiency = used_bits_per_code / bits_per_code
#effective_bitrate = raw_bitrate * efficiency

print(f"Raw Bitrate: {raw_bitrate:.2f} bps")
#print(f"Efficiency: {efficiency:.2f}")
#print(f"Effective Bitrate: {effective_bitrate:.2f} bps")

print(f"SNR: {snr:.2f} dB")
print(f"LSD: {lsd:.4f}")



print("Evaluation complete. Saved as reconstructed_model_b2ae.wav")



