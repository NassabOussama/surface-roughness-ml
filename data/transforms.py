import torchvision.transforms as T


def get_transforms(input_size: int, mode: str = "train") -> T.Compose:
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if mode == "train":
        return T.Compose([
            T.RandomResizedCrop(input_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            T.RandomRotation(180),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.1),
            T.ToTensor(),
            normalize,
        ])
    else:
        return T.Compose([
            T.Resize(int(input_size * 1.1)),
            T.CenterCrop(input_size),
            T.ToTensor(),
            normalize,
        ])
