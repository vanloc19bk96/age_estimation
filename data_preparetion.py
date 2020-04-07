import numpy as np
import os
import logging
from dataset import AgeEstimationDataset


args = {
    "gpuid": 0,
    "image_size": 256,
    "crop_size": 224,
    "is_transform": True
}
def prepare_data(data_path):
    train_data = []
    train_dict = np.load(os.path.join(data_path, "train_cacd_processed.npy"), allow_pickle=True).item()
    logging.info("CACD dataset (training set)")

    val_data = []
    val_dict = np.load(os.path.join(data_path, "valid_cacd_processed.npy"), allow_pickle=True).item()
    logging.info("CACD dataset (valid set)")

    test_data = []
    test_dict = np.load(os.path.join(data_path, "test_cacd_processed.npy"), allow_pickle=True).item()
    logging.info("CACD dataset (test set)")

    train_data.append(AgeEstimationDataset(train_dict, args, 'train'))
    val_data.append(AgeEstimationDataset(val_dict, args, 'val'))
    test_data.append(AgeEstimationDataset(test_dict, args, 'test'))

    return {'train': train_data, 'val': val_data, 'test': test_data}


