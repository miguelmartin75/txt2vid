import torch
import torch.nn as nn

class FrameMap(nn.Module):
    def __init__(self, num_channels=3):
        super().__init__()

        self.frame_map = nn.Sequential(
            nn.Conv2d(num_channels, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(512),
            #nn.LeakyReLU(0.2, True),
        )


    def forward(self, videos):
        frames = videos.permute(0, 2, 1, 3, 4)

        frames_mapped = []
        for i in range(frames.size(1)):
            frame = frames[:, i, :, :, :]
            frame = self.frame_map(frame)
            frames_mapped.append(frame)

        frames = torch.stack(frames_mapped)
        return frames


class FrameDiscrim(nn.Module):
    def __init__(self, txt_encode_size=256):
        super().__init__()

        self.frame_map = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.predictor = nn.Sequential(
            nn.Conv2d(512 + txt_encode_size, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 1, 2, 2, 0, bias=False),
            #nn.LeakyReLU(0.2, True),
            #nn.Linear(3*3, 1)
            #nn.Sigmoid()
        )

        self.sent_map = nn.Sequential(
            nn.Linear(txt_encode_size, txt_encode_size),
            nn.BatchNorm1d(txt_encode_size),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x=None, cond=None, xbar=None):
        sent = self.sent_map(cond)

        outputs = []
        # input is frame x batch x ch x h x w
        # loop through each frame
        for i in range(xbar.size(0)): 
            frame = xbar[i, :, :, :, :]
            frame = self.frame_map(frame)

            sent_dupe = torch.zeros(sent.size(0), sent.size(1), frame.size(2), frame.size(3), device=xbar.device)

            for i in range(frame.size(2)):
                for j in range(frame.size(3)):
                    sent_dupe[:, :, i, j] = sent

            #print('frame=', frame.size())
            #print('sent=', sent_dupe.size())
            frame_and_sent = torch.cat((frame, sent_dupe), dim=1)
            output = self.predictor(frame_and_sent)
            output = output.view(output.size(0), -1).squeeze(1)
            outputs.append(output)

        return torch.stack(outputs, 0).to(xbar.device)

if __name__ == '__main__':
    from txt2vid.util.misc import count_params

    batch_size = 64
    num_channels = 3
    cond_size = 256
    frame_size = 48
    num_frames = 16

    vid = torch.randn(batch_size, num_channels, num_frames, frame_size, frame_size)
    cond = torch.randn(batch_size, cond_size)

    frame_map = FrameMap()
    frames = frame_map(vid)

    print("output (frame_map) =", frames.size())
    print("Num params (frame_map) = %d" % count_params(frame_map))

    discrim = FrameDiscrim()
    out = discrim(x=vid, cond=cond, xbar=frames)

    print("Output frame discrim:", out.size())

    print("Num params = %d" % count_params(discrim))

