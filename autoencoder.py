import torch.nn as nn

class Autoencoder1(nn.Module):
    def __init__(self):
        super(Autoencoder1,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64*64, 1000),
            nn.Linear(1000, 10)
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 1000),
            nn.Linear(1000, 64*64) 
        )        
    def forward(self,x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        out = self.decoder(encoded).view(x.size(0), 1, 64, 64)
        return out

class Autoencoder2(nn.Module):
    def __init__(self):
        super(Autoencoder2,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64*64, 32*32),
            nn.PReLU(32*32),
            nn.Linear(32*32, 16*16),
            nn.PReLU(16*16),
            nn.Linear(16*16,4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 16*16),
            nn.PReLU(16*16),
            nn.Linear(16*16, 32*32),
            nn.PReLU(32*32),
            nn.Linear(32*32, 64*64)
        )   
                
    def forward(self,x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
       # out = self.decoder(encoded)
        out = self.decoder(encoded).view(x.size(0), 1, 64, 64)
        return out

class Autoencoder3(nn.Module):
    def __init__(self):
        super(Autoencoder3,self).__init__()
        self.encoder = nn.Sequential (
          # conv 1
          nn.Conv2d(in_channels= 1, out_channels=16, kernel_size=3, stride=1, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(16),

          # # conv 2
          nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(32),

          # # conv 3
          nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(64),

          # # conv 4
          nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(128),

          # # conv 5
          nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(256)
        )

        self.decoder = nn.Sequential (
          # # conv 6
          nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(128),

          # # conv 7
          nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(64),

          # # conv 8
          nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(32),

          # # conv 9
          nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(16),

          # conv 10
          nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1),
          nn.Tanh()
        )

    def forward(self, x):
      encoded = self.encoder(x)
      out = self.decoder(encoded).view(x.size(0), 1, 64, 64)
      return out


class Autoencoder4(nn.Module):
    def __init__(self):
        super(Autoencoder4,self).__init__()
        self.encoder = nn.Sequential (
          # conv 1
          nn.Conv2d(in_channels= 1, out_channels=16, kernel_size=3, stride=1, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(16),

          # # conv 2
          nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(32),

          # # conv 3
          nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(64),

          # # conv 4
          nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(128),

          # # conv 5
          nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(256)
        )

        self.decoder = nn.Sequential (
          # # conv 6
          nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(128),

          # # conv 7
          nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(64),

          # # conv 8
          nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(32),

          # # conv 9
          nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(16),

          # conv 10
          nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1),
          nn.Sigmoid()
        )

    def forward(self, x):
      encoded = self.encoder(x)
      out = self.decoder(encoded).view(x.size(0), 1, 64, 64)
      return out

class Autoencoder5(nn.Module):
    def __init__(self):
        super(Autoencoder5,self).__init__()
        self.encoder = nn.Sequential (
          # conv 1
          nn.Conv2d(in_channels= 1, out_channels=16, kernel_size=3, stride=3, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(16),

          # # conv 2
          nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=3, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(32),

          # # conv 3
          nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(64)
        )

        self.decoder = nn.Sequential (
          # # conv 4
          nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(32),

          # # conv 5
          nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=3, padding=1),
          nn.PReLU(),
          nn.BatchNorm2d(16),

          # conv 5
          nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=3, padding=1),
          nn.Sigmoid()
        )

    def forward(self, x):
      encoded = self.encoder(x)
      out = self.decoder(encoded).view(x.size(0), 1, 64, 64)
      return out
