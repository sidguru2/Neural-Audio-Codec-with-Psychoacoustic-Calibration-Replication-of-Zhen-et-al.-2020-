import torch
import soundfile as sf

def overlap_add(frames, hop_size=480):
    wav = torch.zeros((frames.size(0)-1)*hop_size + 512)
    for i, frame in enumerate(frames):
        wav[i*hop_size:i*hop_size+512] += frame.squeeze()
    return wav

def save_audio(frames, path="reconstructed.wav", sr=44100, hop_size=480):
    wav = overlap_add(frames, hop_size)
    sf.write(path, wav.numpy(), sr)
