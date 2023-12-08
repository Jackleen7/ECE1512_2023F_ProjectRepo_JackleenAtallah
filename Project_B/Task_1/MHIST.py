# -*- coding: utf-8 -*-
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

#Step 1: Define MHISTDataset Class
class MHISTDataset(Dataset):
    def __init__(self, 
                 csv_file, root_dir, transform=None,partition=None):
        """
        Args:
            csv_file (string): ('C:\\Users\Jackyy\OneDrive - University of Toronto\D\DIP\projectB_sh\DatasetCondensation-master\\mhist_dataset\\annotations.csv')
            root_dir (string): ('C:\\Users\Jackyy\OneDrive - University of Toronto\D\DIP\projectB_sh\DatasetCondensation-master\mhist_dataset\images\images')
            transform (callable, optional): Optional transform to be applied on a sample.
            partition (string, optional):('train' or 'test').
        """
        self.data_frame = pd.read_csv(csv_file)
        if partition:
            self.data_frame = self.data_frame[self.data_frame['Partition'].str.lower() == partition.lower()]
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['HP', 'SSA']

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'images', self.data_frame.iloc[idx]['Image Name'])
        image = Image.open(img_name).convert('RGB')
        label_str = self.data_frame.iloc[idx]['Majority Vote Label']
        if label_str == 'HP':
            label = 0  # Assuming label is an integer
        elif label_str == 'SSA':
            label = 1
        else:
            raise ValueError('Unknown label')
            
        if self.transform:
            image = self.transform(image)

        return image, label
#Step 2: Define get_dataset Function
def get_mhist_dataset(data_path, csv_file, transform=True):
    # Define a default transform if none is provided
    #im_size = (128, 128)
    im_size = (128, 128)
    if transform:
        transform = transforms.Compose([
            transforms.Resize(im_size),  # Resize the images to a common size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7378184, 0.64858836, 0.775252], 
                                 std=[0.16837841, 0.21966821, 0.13939074])
    ])

    # Load the dataset with partitioning
    train_dataset = MHISTDataset(csv_file=os.path.join(data_path, csv_file),
                                 root_dir=data_path, transform=transform, partition='train')
    test_dataset = MHISTDataset(csv_file=os.path.join(data_path, csv_file),
                                root_dir=data_path, transform=transform, partition='test')
    
    channel = 3
    num_classes = 2
    mean=[0.7378184, 0.64858836, 0.775252]
    std=[0.16837841, 0.21966821, 0.13939074]
    class_names = train_dataset.classes
    return channel, im_size, num_classes, class_names, mean, std, train_dataset, test_dataset

#load data 
# data_path = 'C:\\Users\Jackyy\OneDrive - University of Toronto\D\DIP\projectB_sh\DatasetCondensation-master\mhist_dataset'
# ann_file = 'annotations.csv'
# train_dataset, test_dataset = get_mhist_dataset(data_path, ann_file)

# # Create DataLoaders for the datasets
# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers = 4)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # # sample = next(iter(train_loader))
# # # images, labels = sample
# # # image = images[0]  # Get the first image in the batch
# # # label = labels[0]  # Get the corresponding label

# images_train, labels_train = [], []
# for images, labels in test_loader:
#     images_train.append(images.numpy())
#     labels_train.append(labels.numpy())

# # images_train = np.concatenate(images_train, axis = 0)
# # labels_train = np.concatenate(labels_train, axis = 0)
    
# ex_image = images[0].numpy()
# plt.imshow(np.transpose(ex_image, [1,2,0]))
