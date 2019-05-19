import torch
import torch.nn as nn

# assumes 48x48 input
class Gen(nn.Module):
    def __init__(self, z_size=100, cond_dim=0, num_channels=3, scale_factor=1):
        super().__init__()

        self.cond_dim = cond_dim
        self.latent_size = z_size
        self.input_size = self.latent_size + self.cond_dim
        
        self.seq = nn.Sequential(
            nn.ConvTranspose3d(self.input_size, int(512*scale_factor), kernel_size=(2, 6, 6), padding=0, bias=False),
            nn.BatchNorm3d(int(512*scale_factor)),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(int(512*scale_factor), int(256*scale_factor), kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(int(256*scale_factor)),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(int(256*scale_factor), int(128*scale_factor), kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(int(128*scale_factor)),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(int(128*scale_factor), int(64*scale_factor), kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(int(64*scale_factor)),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(int(64*scale_factor), num_channels, kernel_size=1, stride=1, padding=0, bias=False),

            nn.Tanh()
        )

        self.input_map = nn.Sequential(
            nn.Linear(self.input_size, self.input_size),
            nn.BatchNorm1d(self.input_size),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x, cond=None):
        if cond is not None:
            x = torch.cat((x, cond), dim=1)

        x = x.view(x.size(0), x.size(1))
        x = self.input_map(x)
        x = x.view(x.size(0), x.size(1), 1, 1, 1)
        vids = self.seq(x)
        return vids

if __name__ == '__main__':
    # The number of frames in a video is fixed at 16
    batch_size = 8 
    num_channels = 3
    z_size = 100
    cond_size = 256

    gen = Gen(z_size=z_size, num_channels=3, scale_factor=1, cond_dim=cond_size)
    z = torch.randn(batch_size, z_size)
    cond = torch.randn(batch_size, cond_size)
    out = gen(z, cond=cond)
    print("Output video generator:", out.size())
