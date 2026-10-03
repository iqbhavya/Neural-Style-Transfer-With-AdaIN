from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms

class ImageFolderDataset(Dataset):
    def __init__(self, root, transform):

        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform

        self.files = list(os.listdir(root))
        self.files = [f for f in self.files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.files[index])
        image = Image.open(img_path)


        if self.transform:
            image = self.transform(image)

        return image    

def get_transform(size, crop , final_size):

    transform_list = []
    if size is not None:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)    