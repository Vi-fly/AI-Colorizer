import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=3):
        super(UNetGenerator, self).__init__()

        def down_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        def up_block(in_channels, out_channels, dropout=False):
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        # Downsampling
        self.down1 = down_block(input_channels, 64, False)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)
        self.down5 = down_block(512, 512)
        self.down6 = down_block(512, 512)
        self.down7 = down_block(512, 512)
        self.down8 = down_block(512, 512, False)

        # Upsampling
        self.up1 = up_block(512, 512, True)
        self.up2 = up_block(1024, 512, True)
        self.up3 = up_block(1024, 512, True)
        self.up4 = up_block(1024, 256)
        self.up5 = up_block(512, 128)
        self.up6 = up_block(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, output_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Downsampling
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Upsampling with skip connections
        u1 = self.up1(d8)
        u1 = torch.cat([u1, d7], 1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d6], 1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d5], 1)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, d4], 1)
        u5 = self.up5(u4)
        u5 = torch.cat([u5, d3], 1)
        u6 = self.up6(u5)
        u6 = torch.cat([u6, d2], 1)

        return self.final(u6)
    


# -----------------------------------
# 2. PatchGAN Discriminator
# -----------------------------------
class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=4):  # (Gray + Color Image)
        super(PatchGANDiscriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, normalization=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            discriminator_block(input_channels, 64, normalization=False),
            discriminator_block(64, 128),
            discriminator_block(128, 256),
            discriminator_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, gray_img, color_img):
        combined = torch.cat([gray_img, color_img], dim=1)
        return self.model(combined)


# -----------------------------------
# Testing the Models (Optional)
# -----------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test Generator
    generator = UNetGenerator().to(device)
    input_tensor = torch.randn(1, 1, 256, 256).to(device)  # 1 grayscale image
    output_tensor = generator(input_tensor)
    print(f"Generator Output Shape: {output_tensor.shape}")  # Should be (1, 3, 256, 256)

    # Test Discriminator
    discriminator = PatchGANDiscriminator().to(device)
    gray_tensor = torch.randn(1, 1, 256, 256).to(device)  # 1 grayscale image
    color_tensor = torch.randn(1, 3, 256, 256).to(device)  # 1 color image
    output_disc = discriminator(gray_tensor, color_tensor)
    print(f"Discriminator Output Shape: {output_disc.shape}")  # Should be (1, 1, 30, 30)
