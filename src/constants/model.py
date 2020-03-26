"""
Constants for eye2gene project
"""
from torchvision import datasets, models, transforms

DATA_TRANSFORMS = {
    "train": transforms.Compose([
        transforms.CenterCrop(1152),
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]),
    "finetune": transforms.Compose([
        transforms.CenterCrop(1152),
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])
}

data_transforms_old = {
    'train': transforms.Compose([
        transforms.CenterCrop(1536),
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        # ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'finetune': transforms.Compose([
        transforms.CenterCrop(1536),
        transforms.Resize((299, 299)),
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_transforms_pad = {
        'train': transforms.Compose([
            # transforms.Pad((0, pad_size)),
            transforms.Pad((0, 192)),
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),
        'finetune': transforms.Compose([
            # transforms.Pad((0, pad_size)),
            transforms.Pad((0, 192)),
            transforms.Resize((299, 299)),
            transforms.ToTensor()
            ]),
}

data_transforms_cropped = {
        'train': transforms.Compose([
            # transforms.CenterCrop(crop_size),
            transforms.CenterCrop(1536),
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),
        'finetune': transforms.Compose([
            # transforms.CenterCrop(crop_size),
            transforms.CenterCrop(1536),
            transforms.Resize((299, 299)),
            transforms.ToTensor()
            ]),
    }

DATA_TRANSFORMS_TRAIN = {
    "train": transforms.Compose([
        transforms.CenterCrop(1536),
        transforms.RandomResizedCrop(299),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(90),
        transforms.ToTensor()
    ]),
    "finetune": transforms.Compose([
        transforms.CenterCrop(1536),
        transforms.Resize((299,299)),
        transforms.ToTensor()
    ])
}