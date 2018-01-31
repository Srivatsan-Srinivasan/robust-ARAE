class HRFConvnet(t.nn.Module):
    def __init__(self):
        super(HRFConvnet, self).__init__()
        self.conv1a = conv(300, 25, 3, padding=1)
        self.conv1b = conv(300, 25, 5, padding=2)
        self.conv2 = conv(100, 50, 2, padding=1)
        self.maxpool = t.nn.MaxPool1d(3, padding=1)
        self.avgpool = t.nn.AvgPool1d(3, padding=1)
        self.dropout = Dropout(.25)
        self.fc2 = fc(200, 2)

    def forward(self, x):
        # convolutions and pooling
        xx = lrelu(t.cat([self.conv1a(x), self.conv1b(x)], 1))
        xx = t.cat([self.maxpool(xx), self.avgpool(xx)], 1)
        xx = lrelu(self.conv2(xx))

        # several kinds of pooling over time
        xx_max = t.max(xx, -1)[0]
        xx_mean = t.mean(xx, -1)
        xx_min = t.min(xx, -1)[0]
        xx_med = t.median(xx, -1)[0]
        xx = t.cat([xx_max, xx_mean, xx_min, xx_med], -1)

        # dropout and linear layer
        xx = self.dropout(xx)
        xx = self.fc2(xx)
        return softmax(xx)


class SHRFConvnet(t.nn.Module):
    def __init__(self):
        super(SHRFConvnet, self).__init__()
        self.conv1a = conv(300, 25, 3, padding=1)
        self.conv1b = conv(300, 25, 5, padding=2)
        self.conv2 = conv(50, 50, 2, padding=1)
        self.maxpool = t.nn.MaxPool1d(3, padding=1)
        self.dropout = Dropout(.25)
        self.fc2 = fc(50, 2)

    def forward(self, x):
        # convolutions and pooling
        xx = lrelu(t.cat([self.conv1a(x), self.conv1b(x)], 1))
        xx = self.maxpool(xx)
        xx = lrelu(self.conv2(xx))

        # several kinds of pooling over time
        xx = t.max(xx, -1)[0]

        # dropout and linear layer
        xx = self.dropout(xx)
        xx = self.fc2(xx)
        return softmax(xx)