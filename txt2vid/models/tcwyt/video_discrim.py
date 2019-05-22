import torch
import torch.nn as nn

class VideoDiscrim(nn.Module):

    def __init__(self, txt_encode_size=256, num_filters=64, num_channels=3):
        super().__init__()

        self.vid = nn.Sequential(
            nn.Conv3d(num_channels, 64, 4, 2, 1, bias=False), # 64
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv3d(64, 128, 4, 2, 1, bias=False), # 128
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv3d(128, 256, 4, 2, 1, bias=False), 
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv3d(256, 512, 4, 2, 1, bias=False), # 256
            nn.BatchNorm3d(512),
            #nn.LeakyReLU(0.2, True),
            # ===


            #nn.BatchNorm3d(1024),
            #nn.LeakyReLU(0.2, True),

            #nn.Conv3d(1024, txt_encode_size, (2, 4, 4), (1, 1, 1), 0, bias=False), # 512
            #nn.BatchNorm3d(txt_encode_size),
            #nn.LeakyReLU(0.2, True),
        )

        self.sent_map = nn.Sequential(
            nn.Linear(txt_encode_size, txt_encode_size),
            nn.BatchNorm1d(txt_encode_size),
            nn.LeakyReLU(0.2, True)
        )

        self.predictor = nn.Sequential(
            nn.Conv3d(512 + txt_encode_size, 512, (1,1,1), 1, 0, bias=False),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(512, 1, (1, 2, 2), (1, 2, 2), 0, bias=False),
            #nn.LeakyReLU(0.2, True),
            #nn.Linear(3*3, 1)
            #nn.Sigmoid()
        )

    def forward(self, x=None, cond=None, xbar=None):
        sent = self.sent_map(cond)
        vids = self.vid(x)

        sent_temp = torch.zeros((vids.size(0), sent.size(1), vids.size(2), vids.size(3), vids.size(4)), device=x.device)
        for i in range(vids.size(2)):
            for j in range(vids.size(3)):
                for k in range(vids.size(4)):
                    sent_temp[:, :, i, j, k] = sent

        sent = sent_temp
        vids_plus_sent = torch.cat((vids, sent), dim=1)

        pred = self.predictor(vids_plus_sent)
        pred = pred.view(-1, 1).squeeze(1)
        return pred

if __name__ == '__main__':
    batch_size = 64
    num_channels = 3
    cond_size = 256
    frame_size = 48
    num_frames = 16

    vid = torch.randn(batch_size, num_channels, num_frames, frame_size, frame_size)
    cond = torch.randn(batch_size, cond_size)

    discrim = VideoDiscrim()
    out = discrim(x=vid, cond=cond)

    print("Output video discrim:", out.size())

    from txt2vid.util.misc import count_params
    print("Num params = %d" % count_params(discrim))

