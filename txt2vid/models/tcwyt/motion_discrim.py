import torch
import torch.nn as nn

class MotionDiscrim(nn.Module):
    def __init__(self, cond_dim=256):
        super().__init__()

        self.motion_map = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.predictor = nn.Sequential(
            nn.Conv2d(512 + cond_dim, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 1, 2, 2, 0, bias=False),
            #nn.LeakyReLU(0.2, True),
            #nn.Linear(3*3, 1)
            #nn.Sigmoid()
        )

        self.sent_map = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.BatchNorm1d(cond_dim),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x=None, cond=None, xbar=None):
        sent = self.sent_map(cond)
        motions = xbar[1:] - xbar[0:-1]

        outputs = []
        # loop through each motion 'frame'
        for i in range(motions.size(0)): 
            motion = motions[i, :, :, :, :]
            motion = self.motion_map(motion)

            sent_dupe = torch.zeros((sent.size(0), sent.size(1), motion.size(2), motion.size(3)), device=xbar.device)

            for i in range(motion.size(2)):
                for j in range(motion.size(3)):
                    sent_dupe[:, :, i, j] = sent

            motion_and_sent = torch.cat((motion, sent_dupe), dim=1)
            output = self.predictor(motion_and_sent)
            output = output.view(output.size(0), -1).squeeze(1)
            outputs.append(output)

        return torch.stack(outputs, 0).to(xbar.device)

if __name__ == '__main__':
    from txt2vid.util.misc import count_params
    from txt2vid.models.tcwyt.frame_discrim import FrameMap

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

    discrim = MotionDiscrim()
    out = discrim(x=vid, cond=cond, xbar=frames)

    print("Output frame discrim:", out.size())

    print("Num params = %d" % count_params(discrim))

