from torchvision import transforms

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
data_transform = transforms.Compose([  # Accuracy:87.622%

        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
data_transform1 = transforms.Compose([ # Accuracy:83.263%

        transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
data_transform2 = transforms.Compose([ # Accuracy:85.524%
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
data_transform3 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

data_transform4 = transforms.Compose([  #

        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
