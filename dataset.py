import os
import json

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
def get_dataloader(dataset_dir, batch_size=1, split='train'):
    ###############################
    # TODO:                       #
    # Define your own transforms. #
    ###############################
    if split == 'train':
        transform = transforms.Compose([

            ##### TODO: Data Augmentation Begin #####
            # transforms.RandomRotation(45, resample=False, expand=False, center=None, fill=None),
            # others https://chih-sheng-huang821.medium.com/03-pytorch-dataaug-a712a7a7f55e
            ##### TODO: Data Augmentation End #####
            transforms.RandomHorizontalFlip(p=0.3),    # half rotate 
            transforms.ColorJitter(saturation=0.1),
            transforms.GaussianBlur(3, sigma=(1)),
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else: # 'val' or 'test'
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            # we usually don't apply data augmentation on test or val data
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = CIFAR10Dataset(dataset_dir, split=split, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), num_workers=0, pin_memory=True, drop_last=(split=='train'))

    return dataloader

class CIFAR10Dataset(Dataset):
    def __init__(self, dataset_dir, split='test', transform=None):
        super(CIFAR10Dataset).__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform

        with open(os.path.join(self.dataset_dir, 'annotations.json'), 'r') as f:
            json_data = json.load(f)
            

        self.image_names = json_data['filenames']
        # print("image_names",self.image_names[0])
        if self.split != 'test':
            self.labels = json_data['labels']
        # print("labels",self.labels[0])
        print(f'Number of {self.split} images is {len(self.image_names)}')
        # print()
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):

        ########################################################
        # TODO:                                                #
        # Define the CIFAR10Dataset class:                     #
        #   1. use Image.open() to load image according to the # 
        #      self.image_names                                #
        #   2. apply transform on image                        #
        #   3. if not test set, return image and label with    #
        #      type "long tensor"                              #
        #   4. else return image only                          #
        #                                                      #
        # NOTE:                                                #
        # You will not have labels if it's test set            #
        ########################################################
        convert_tensor = transforms.ToTensor()
        read_image=Image.open(os.path.join(self.dataset_dir,self.image_names[index]))
        image=convert_tensor(read_image)
        # label=torch.tensor(self.labels[index], dtype=torch.long) 
        if self.split != 'test':
            label=torch.tensor(self.labels[index])
            return {
                'images': image, 
                'labels': label
            }            
        else:
            return {
                'images': image, 
            }
