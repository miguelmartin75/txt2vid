class MotionDiscrim(nn.Module):
    def __init__(self, txt_encode_size=256):
        super().__init__()

        self.motion_map = nn.Sequential(
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

        self.apply(weights_init)

    def forward(self, frames, sent=None, device=None):
        sent = self.sent_map(sent)
        motions = frames[1:] - frames[0:-1]

        outputs = []
        # loop through each motion 'frame'
        for i in range(motions.size(0)): 
            motion = motions[i, :, :, :, :]
            motion = self.motion_map(motion)

            sent_dupe = torch.zeros((sent.size(0), sent.size(1), motion.size(2), motion.size(3)), device=device)

            for i in range(motion.size(2)):
                for j in range(motion.size(3)):
                    sent_dupe[:, :, i, j] = sent

            motion_and_sent = torch.cat((motion, sent_dupe), dim=1)
            output = self.predictor(motion_and_sent)
            output = output.view(output.size(0), -1).squeeze(1)
            outputs.append(output)

        return torch.stack(outputs, 0).to(device)
