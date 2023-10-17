import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt

from datasets.transforms import build_transform
from test.datasets import build_dataset_ccrop
from test.datasets.transforms import cifar_train_ccrop

if __name__ == '__main__':
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    root = './data'
    crop_dict = dict(
        type='cifar_train_ccrop',
        alpha=0.1,
        mean=mean, std=std
    )
    ds_dict = dict(
        type='CIFAR10_boxes',
        root=root,
        train=True,
    )
    rcrop_dict = dict(
        type='cifar_train_rcrop',
        mean=mean, std=std
    )

    transform = cifar_train_ccrop(alpha=0.6, mean=mean, std=std)

    # Load the CIFAR-10 dataset
    # train_set = build_dataset_ccrop(crop_dict, ds_dict, rcrop_dict)
    # train_loader = torch.utils.data.DataLoader(
    #     train_set,
    #     batch_size=1,
    #     num_workers=1,
    #     pin_memory=True,
    #     drop_last=True
    # )
    # for idx, (images, labels) in enumerate(train_loader):
    #     if idx == 1:
    #         break
    #     print(labels)
    #     a = images[0].permute(0,2,3,1)
    #     k = a.cpu().detach().numpy()
    #     image = Image.fromarray(np.uint8(k[0])).convert('RGB')
    #     image.show()


    image = Image.open('dog.jpg')
    box = (0., 0., 1., 1.)
    img = transform([image, box])
    images = torch.stack([img[1]], 1)
    images = images.view(-1, 3, 32, 32)
    # print(images.shape)
    def transform_invert(img, show=True):
        # Tensor -> PIL.Image
        # 注意：img.shape = [3,32,32] cifar10中的一张图片，经过transform后的tensor格式

        if img.dim() == 3:  # single image # 3,32,32
            img = img.unsqueeze(0)  # 在第0维增加一个维度 1,3,32,32
        low = float(img.min())
        high = float(img.max())
        # img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))  # (img - low)/(high-low)
        grid = img.squeeze(0)  # 去除维度为1的维度
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        if show:
            img.show()
        return img
    transform_invert(img[0])


    # plt.imshow(img)
    # plt.show()

    # # Get the first image and its corresponding label
    # image, label = cifar_dataset[9]
    #
    # # Convert the tensor image to numpy array and transpose it
    # image = transformed_image.permute(1, 2, 0).numpy()
    #
    # # Display the image
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()