import torch
import torchaudio
import os
from glob import glob
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import torchaudio.transforms as T


# Parameters
data_root = "data/genres_original"
all_wav_paths = glob(os.path.join(data_root, "*", "*.wav"))
frame_size = 512
hop_size = 480
hann_window = torch.hann_window(frame_size)
def make_partial_hann_window(frame_size=512, overlap=32):
    window = torch.ones(frame_size)
    hann_overlap = torch.hann_window(overlap*2)

    window[:overlap] = hann_overlap[:overlap]  # fade in
    window[-overlap:] = hann_overlap[overlap:]  # fade out

    return window

partial_hann_window = make_partial_hann_window(frame_size=512, overlap=32)


def frame_audio(wav, frame_size=512, hop_size=480, window=None):
    frames = wav.unfold(1, frame_size, hop_size).squeeze(0)
    if window is not None:
        frames = frames * window  # apply window to each frame
    return frames

all_frames = []

for path in all_wav_paths:
    waveform, sr = torchaudio.load(path)
    if sr != 44100:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=44100)
    waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform / waveform.abs().max()

    if waveform.shape[1] < frame_size:
        print(f"Skipped (too short): {path}")
        continue

    frames = frame_audio(waveform, frame_size, hop_size, partial_hann_window)

    if frames.shape[1] != frame_size:
        print(f"Bad frame size: {frames.shape}")
        continue

    all_frames.extend(frames)

print(f"Total frames collected: {len(all_frames)}")


class FrameDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx].unsqueeze(0)  # shape: (1, 512)

dataset = FrameDataset(all_frames)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


from model.modules import Encoder, Decoder, SoftToHardQuantizer
from utils.losses import sse_loss, MelLoss
from utils.audio_io import save_audio
model_type = '1AE' #or 1AE

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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

quantizer = SoftToHardQuantizer(num_kernels=64, alpha=300).to(device)
if model_type == '1AE':
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(quantizer.parameters()),
        lr=2e-4
    )
else:  # 2AE
    optimizer = torch.optim.Adam([
        {'params': encoder1.parameters(), 'lr': 2e-4},
        {'params': decoder1.parameters(), 'lr': 2e-4},
        {'params': quantizer.parameters(), 'lr': 2e-4},
        {'params': encoder2.parameters(), 'lr': 2e-5},
        {'params': decoder2.parameters(), 'lr': 2e-5},
    ])


#num_epochs = 1

from utils.losses import sse_loss, MelLoss, PriorityWeightingLossPAM1

# training parameters
num_epochs = 20
patience = 3
min_delta = 1e-6

best_loss = float('inf')
epochs_without_improvement = 0

λ2 = 0.1
mel_loss_fn = MelLoss(sample_rate=44100, n_fft=512, hop_length=480, n_mels=40).to(device)
priority_weighting_loss = PriorityWeightingLossPAM1(sample_rate=44100, n_fft=512, hop_length=480).to(device)

for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        batch = batch.to(device)

        # === Forward pass ===
        if model_type == '1AE':
            z = encoder(batch)
            z_q, _ = quantizer(z)
            decoded = decoder(z_q)
            #print(decoded.shape)
        elif model_type == '2AE':
            z1 = encoder1(batch)
            z1_q, _ = quantizer(z1)
            x1_hat = decoder1(z1_q)
            residual = batch - x1_hat
            z2 = encoder2(residual)
            z2_q, _ = quantizer(z2)
            residual_hat = decoder2(z2_q)
            decoded = x1_hat + residual_hat
            


       
        loss_sse = sse_loss(decoded, batch)
        loss_mel = mel_loss_fn(decoded, batch)
        loss_l3 = priority_weighting_loss(decoded, batch)
        loss = loss_sse + λ2 * (loss_mel + loss_l3)
        if batch_idx < 5:  # debug prints
#            print(f"\n=== Debug Logs 2ae(Batch {batch_idx}) ===")
#            print("Stage 1: z1 std: {:.5f}, z1_q std: {:.5f}".format(z1.std().item(), z1_q.std().item()))
#            print("Stage 2: z2 std: {:.5f}, z2_q std: {:.5f}".format(z2.std().item(), z2_q.std().item()))
#
#            print("x1_hat → max: {:.5f}, min: {:.5f}, std: {:.5f}".format(
#                x1_hat.max().item(), x1_hat.min().item(), x1_hat.std().item()))
#            print("residual_hat → max: {:.5f}, min: {:.5f}, std: {:.5f}".format(
#                residual_hat.max().item(), residual_hat.min().item(), residual_hat.std().item()))
#            print("decoded → max: {:.5f}, min: {:.5f}, std: {:.5f}".format(
#                decoded.max().item(), decoded.min().item(), decoded.std().item()))
#
#            print("Loss SSE: {:.5f}, Mel Loss: {:.5f}".format(loss_sse.item(), loss_mel.item()))
#
#            print("Total: {:.5f}".format(loss.item()))

            print(f"Debug Logs 1ae(Batch {batch_idx})")
        
            print("z std: {:.5f}, z_q std: {:.5f}".format(z.std().item(), z_q.std().item()))
            print("decoded max: {:.5f}, min: {:.5f}, std: {:.5f}".format(
                decoded.max().item(), decoded.min().item(), decoded.std().item()))
            print("Loss SSE: {:.5f}, Mel Loss: {:.5f}".format(loss_sse.item(), loss_mel.item()))
#            print("Loss SSE: {:.5f}, Mel Loss: {:.5f}, L3: {:.5f}, Total: {:.5f}".format(
#                loss_sse.item(), loss_mel.item(), loss_l3.item(), (loss_sse + λ2 * loss_mel).item()))

        if torch.isnan(loss):
            print("NaNs in loss! Skipping this batch.")
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 2000 == 0:
#            print(f"\n=== Batch {batch_idx} ===")
#            print("Batch Loss: {:.7f}".format(loss.item()))
#            print("Loss SSE: {:.5f}, Mel Loss: {:.5f}".format(loss_sse.item(), loss_mel.item()))
#            try:
#                print("L3: {:.5f}".format(loss_l3.item()))
#            except:
#                pass
#            print("Decoded → max: {:.5f}, min: {:.5f}".format(decoded.max().item(), decoded.min().item()))
#            print("z1 std: {:.5f}, z1_q std: {:.5f} | z2 std: {:.5f}, z2_q std: {:.5f}".format(
#                z1.std().item(), z1_q.std().item(), z2.std().item(), z2_q.std().item()))
            print(f"Batch Loss: {loss.item():.7f}")
            print("Loss SSE: {:.5f}, Mel Loss: {:.5f}".format(loss_sse.item(), loss_mel.item()))
#            print("Loss SSE: {:.5f}, Mel Loss: {:.5f}, L3: {:.5f}, Total: {:.5f}".format(loss_sse.item(), loss_mel.item(), loss_l3.item(), (loss_sse + λ2 * loss_mel).item()))

            print("Decoded max:", decoded.max().item(), "min:", decoded.min().item())
            print("z std: {:.5f}, z_q std: {:.5f}".format(z.std().item(), z_q.std().item()))
            
            

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.10f}")

   
    if best_loss - avg_loss > min_delta:
        best_loss = avg_loss
        epochs_without_improvement = 0

        if model_type == '1AE':
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'quantizer': quantizer.state_dict(),
            }, "modelc_1ae_weights.pth")
        else:
            torch.save({
                'encoder1': encoder1.state_dict(),
                'decoder1': decoder1.state_dict(),
                'encoder2': encoder2.state_dict(),
                'decoder2': decoder2.state_dict(),
                'quantizer': quantizer.state_dict(),
            }, "modelb_2ae_weights3.pth")

    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

