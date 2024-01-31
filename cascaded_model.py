import torch
from torch.nn import Conv2d, LeakyReLU, Sequential, Module, ReLU


def to_re_img(batch):  # changes complex into re_img
    b, c, h, w = batch.shape
    empty_batch = torch.empty((b, c * 2, h, w))
    re_img, im_img = torch.real(batch), torch.imag(batch)
    empty_batch[:, ::2, :, :] = re_img
    empty_batch[:, 1::2, :, :] = im_img
    return empty_batch.to("cuda:0")


def to_complex(batch):
    batch = batch[:, ::2, :, :] + 1j * batch[:, 1::2, :, :]
    return batch


class DataConsistencyLayer(Module):
    """Makes sure known k-space points are maintained"""

    def __init__(self):
        super().__init__()

    def forward(self, x, og_kspace, mask):
        x = to_complex(x)
        mod_x_kspace = og_kspace + torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1))), dim=(-2, -1)
        ) * (
            1.0 - mask
        )  # x is standard image
        mod_x = torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(mod_x_kspace, dim=(-2, -1))),
            dim=(-2, -1),
        )
        mod_x = to_re_img(mod_x)
        return mod_x


class CascadeBlock(Module):
    def __init__(self, input_channels, block_depth, filters) -> None:
        super().__init__()

        layers = []
        layers.append(Conv2d(input_channels, filters, kernel_size=3, padding="same"))
        layers.append(LeakyReLU())

        for i in range(block_depth - 2):
            layers.append(Conv2d(filters, filters, kernel_size=3, padding="same"))
            layers.append(LeakyReLU())

        layers.append(Conv2d(filters, input_channels, kernel_size=3, padding="same"))
        layers.append(LeakyReLU())

        self.layers = Sequential(*layers)
        self.dc = DataConsistencyLayer()

    def forward(self, info_tuple) -> torch.Tensor:
        x, og_kspace, mask, smap = info_tuple
        x = self.layers(x)
        # do dc layer
        x = self.dc(x, og_kspace, mask)

        return (x, og_kspace, mask, smap)


class CascadedModel(Module):
    def __init__(self, input_channels, reps=5, block_depth=5, filters=110) -> None:
        super().__init__()
        # can iterate through a block or go through layers
        blocks = []
        for i in range(reps):
            blocks.append(CascadeBlock(input_channels, block_depth, filters))
        self.blocks = Sequential(*blocks)

    # batch is batchsize, channel, h, w
    def forward(self, info_tuple):  # info tuple is img input, kspace, mask, smap
        x, _, _, smap = self.blocks(info_tuple)

        x, smap = to_complex(x), to_complex(smap)
        weights = torch.sum(smap * torch.conj(smap), dim=1, keepdim=True)
        sc = to_re_img(torch.sum(x * torch.conj(smap), dim=1, keepdim=True) / weights)

        return sc


def test():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    # B, C, H, W
    input_img = torch.randn(size=(1, 8, 256, 256), dtype=torch.float32).to(
        device
    )  # size 1, 2, 256, 256
    ref_kspace = torch.complex(
        torch.randn(size=(1, 4, 256, 256)), torch.randn(size=(1, 4, 256, 256))
    ).to(device)
    mask = torch.randn(size=(1, 4, 256, 256), dtype=torch.float32).to(device)

    input_channels = 8
    model = CascadedModel(input_channels).type(torch.float32).to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    alt_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable params: {pytorch_total_params}")

    input_tuple = (input_img, ref_kspace, mask)
    img_output, smap_output = model(input_tuple)
    print(input_img.shape, img_output.shape, smap_output.shape)


if __name__ == "__main__":
    test()
