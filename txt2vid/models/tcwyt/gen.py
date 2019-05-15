# assumes 48x48 input
class Generator(nn.Module):
    def __init__(self, latent_size=256, num_channels=1):
        super().__init__()

        self.latent_size = latent_size
        
        self.seq = nn.Sequential(
            nn.ConvTranspose3d(latent_size, 512, kernel_size=(2, 6, 6), padding=0, bias=False),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(64, num_channels, kernel_size=1, stride=1, padding=0, bias=False),

            nn.Tanh()
        )

        self.input_map = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.LeakyReLU(0.2, True)
        )

        self.apply(weights_init)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1))
        x = self.input_map(x)
        x = x.view(x.size(0), x.size(1), 1, 1, 1)
        vids = self.seq(x)
        return vids

if __name__ == '__main__':
    # The number of frames in a video is fixed at 16
    batch_size = 8
    num_channels = 3
    latent_size = 256
    gen = Gen(latent_size=latent_size)
    z = torch.randn(batch_size, latent_size)
    out = gen(z)
    print("Output video generator:", out.size())
