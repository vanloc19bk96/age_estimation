import torch
import logging
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as transforms_f
import PIL
import numpy as np
import itertools

class AgeEstimationDataset(Dataset):
    def __init__(self, dictionary, args, mode):
        self.dictinary = dictionary
        self.image_size = args["image_size"]
        self.crop_size = args["crop_size"]
        self.crop_limit = self.image_size - self.crop_size
        self.is_transform = args["is_transform"]
        self.mode = mode
        self.image_path_list = []
        self.labels = []
        self.scale_factor = 100
        self.mean = [0.432, 0.359, 0.320]
        self.std = [0.30,  0.264,  0.252]
        for key in dictionary:
            self.image_path_list.append(dictionary[key]['path'])
            self.labels.append(dictionary[key]['age_list'])
        # convert to 1D
        self.image_path_list = sum(self.image_path_list, [])
        self.labels = sum(self.labels, [])
        logging.info('{:s} contains {:d} images'.format('CACD', len(self.image_path_list)))
        self.labels = torch.FloatTensor(self.labels)
        self.labels /= self.scale_factor

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        image = PIL.Image.open(image_path)

        # augmentation
        if self.is_transform:
            if np.random.rand() > 0.5 and self.mode != 'test':
                image = transforms_f.hflip(image)
            if self.crop_limit > 1:
                # random cropping
                if self.mode == 'train':
                    x_start = int(self.crop_limit*np.random.rand())
                    y_start = int(self.crop_limit*np.random.rand())
                else:
                    # only apply central crop for evaluation set
                    x_start = 15
                    y_start = 15

                image = transforms_f.crop(image, y_start, x_start, self.crop_size, self.crop_size)

            image = transforms_f.to_tensor(image)
            image = transforms_f.normalize(image, mean = self.mean, std = self.std)

            sample = {'image': image,
                      'age': self.labels[index],
                      'index': index}
            return sample



