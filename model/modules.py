        
import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckBlock(nn.Module):
    def __init__(self, in_ch, bottleneck_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, bottleneck_ch, 9, padding=4), #compress
            nn.ReLU(),
            nn.Conv1d(bottleneck_ch, bottleneck_ch, 9, padding=4), #extract
            nn.ReLU(),
            nn.Conv1d(bottleneck_ch, out_ch, 9, padding=4), #retreive
        )
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.block(x) + self.shortcut(x) #resnet

#class Encoder(nn.Module):
#    def __init__(self, ldim=8):
#        super().__init__()
#        self.enc = nn.Sequential(
#            nn.Conv1d(1, 100, 9, padding=4), #change channel
#            BottleneckBlock(100, 20, 100), #1st BottleNeck
#            nn.Conv1d(100, 100, 9, stride=2, padding=4), #Downsampling
#            BottleneckBlock(100, 20, 100), #2nd BottleNeck
#            nn.Conv1d(100, ldim, 9, padding=4) # change channel, changed 1 to dim to output batch, 8 vectors per frame
#        )
#
#    def forward(self, x):
#        z = self.enc(x) #batch, dim, time
#        
#        return z.mean(dim=2)
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, 100, 9, padding=4),             # (512, 1) → (512, 100)

            BottleneckBlock(100, 20, 100),               # 1st Bottleneck ×2
            BottleneckBlock(100, 20, 100),

            nn.Conv1d(100, 100, 9, stride=2, padding=4), # Downsample: (512 → 256)

            BottleneckBlock(100, 20, 100),               # 2nd Bottleneck ×2
            BottleneckBlock(100, 20, 100),

            nn.Conv1d(100, 1, 9, padding=4)              # Final: Change channel to 1
        )

    def forward(self, x):
        return self.enc(x)


#class Decoder(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.dec = nn.Sequential(
#            nn.Conv1d(1, 100, 9, padding=4),
#            BottleneckBlock(100, 20, 100),
#            nn.ConvTranspose1d(100, 50, 9, stride=2, padding=4, output_padding=1),
#            BottleneckBlock(50, 20, 50),
#            nn.Conv1d(50, 1, 9, padding=4)
#        )
#
#    def forward(self, x):
#        return self.dec(x)
#class Decoder(nn.Module):
#    def __init__(self, ldim=8):
#        super().__init__()
#        self.fc = nn.Linear(ldim, 1 * 256)  # Upsample to (1, 256)
#        self.dec = nn.Sequential(
#            nn.Conv1d(1, 100, 9, padding=4),
#            BottleneckBlock(100, 20, 100),
#            nn.ConvTranspose1d(100, 50, 9, stride=2, padding=4, output_padding=1),
#            BottleneckBlock(50, 20, 50),
#            nn.Conv1d(50, 1, 9, padding=4)
#        )
#
#    def forward(self, x):  # x shape: (batch, latent_dim)
#        x = self.fc(x).view(-1, 1, 256)  # -> (batch, 1, 256)
#        return self.dec(x)               # -> (batch, 1, 512)
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Conv1d(1, 100, kernel_size=9, padding=4),  # create 100 channels, feature reps

            BottleneckBlock(100, 20, 100),                # 1st bottleneck ×2
            BottleneckBlock(100, 20, 100),

            nn.ConvTranspose1d(100, 50, kernel_size=9, stride=2, padding=4, output_padding=1),  # 256, 100 → (512, 50

            BottleneckBlock(50, 20, 50),                  # 2nd bottleneck ×2
            BottleneckBlock(50, 20, 50),

            nn.Conv1d(50, 1, kernel_size=9, padding=4),     # single channel outpout
            #nn.Tanh() #added for model b
        )

    def forward(self, x):
        return self.dec(x)


#
#class SoftToHardQuantizer(nn.Module):
#    def __init__(self, num_kernels=64, alpha=10.0):
#        super().__init__()
#        self.num_kernels = num_kernels
#        self.alpha = alpha
#
#        # codebook - random kernels
#        self.beta = nn.Parameter(torch.randn(num_kernels))  # shape: (K,)
#
#    def forward(self, z, hard=False):
#        
#        #z: shape (batch, code_dim) — assigned to nearest kernel
#        
#        #  (batch * code_dim, 1)
#        z_flat = z.view(-1, 1)
#
#        # get distances
#        dists = (z_flat - self.beta.view(1, -1)) ** 2
#
#        
#        a = F.softmax(-self.alpha * dists, dim=1)
#
#        if hard:
#            # one-hot vector of maximum element
#            idx = torch.argmax(a, dim=1)
#            a_hard = F.one_hot(idx, num_classes=self.num_kernels).float()
#            a = a_hard.detach() + a - a.detach()
#
#        
#        z_hat_flat = torch.matmul(a, self.beta)
#
#        
#        z_hat = z_hat_flat.view(z.shape)
#        return z_hat, a

class SoftToHardQuantizer(nn.Module):
    def __init__(self, num_kernels=64, alpha=10.0):
        super().__init__()
        self.num_kernels = num_kernels
        self.alpha = alpha

        
        self.beta = nn.Parameter(torch.randn(num_kernels))

    def forward(self, z, hard=False):
        

        z_flat = z.view(-1, 1)
        beta = self.beta.view(1, -1)

        dists = (z_flat - beta) ** 2
        a = F.softmax(-self.alpha * dists, dim=1)

        if hard:
            idx = torch.argmax(a, dim=1)
            a_hard = F.one_hot(idx, num_classes=self.num_kernels).float()
            a = a_hard.detach() + a - a.detach()

        z_hat_flat = torch.matmul(a, self.beta)
        z_hat = z_hat_flat.view(z.shape)       
        return z_hat, a
