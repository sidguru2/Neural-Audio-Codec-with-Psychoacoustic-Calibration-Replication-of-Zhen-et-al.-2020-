import torch.nn.functional as F
from model.modules import SoftToHardQuantizer
quantize = SoftToHardQuantizer(num_kernels=64, alpha=10.0)
for epoch in range(num_epochs):
    for frames in dataloader:
        x = frames.to(device) 
        z = encoder(x)
        h, _ = quantize(z)
        x_hat = decoder(h)
        loss = sse_loss(x_hat, x) #optional quantization error
        #quant_error = F.mse_loss(z_q, z.detach())
        #loss = loss + 0.1 * quant_error
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
