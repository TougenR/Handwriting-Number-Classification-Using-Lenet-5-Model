import torch
import torch.nn as nn

class NumberClassification(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.extract_feature1 = self.make_layer(1,6)
        self.extract_feature2 = self.make_layer(6,16)
        self.extract_feature3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(in_features=84, out_features=num_classes)

    def make_layer(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.extract_feature1(x)
        x = self.extract_feature2(x)
        x = self.extract_feature3(x)

        # flatten
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x) 

        return x

if __name__ == "__main__":
    model = NumberClassification()
    sample = torch.rand(1, 1, 32, 32)
    output = model(sample)
    print(output)
    print(torch.argmax(output, dim=1))