import torch
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import v2
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
#import pytorch_lightning as pl
#from torchvision.transforms.v2 import RandomHorizontalFlip, RandomVerticalFlip
#import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import pickle
from torch.utils.data import default_collate
import random
#from utils.data_manipulation import TioNormalizeStandard
import nibabel as nib
import numpy as np
from monai import transforms as t
from monai.apps import datasets

#%%
#dataset = datasets.TciaDataset(root_dir='/home/brandtj/Documents/projects/iderha/nlst/binary_classifier/decoy', download=True, collection='NLST', download_len=5, section='train', seg_type='SEG')



class Nodule_classifier_online_extraction(Dataset):
    def __init__(self, pickle_path, volumetric, windowed,transform=None, crop_size=(64,64,32)):
        self.pickle_path = pickle_path
        self.transform = transform
        self.volumetric = volumetric
        self.windowed = windowed
        self.mean = 0.11195465258950157
        self.std_dev = 0.06940309561384332
        self.crop_size = crop_size
        self.root_masks = '/home/brandtj/Documents/data/NLST/data/lungs_seg'
        #self.time_points = ['T0', 'T1', 'T2']

        # Load the pickled dataset
        with open(pickle_path, 'rb') as pickle_file:
            self.data_dict = pickle.load(pickle_file)
        self.data_keys = [key for key in self.data_dict.keys()]

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        subject = self.data_keys[idx]

        #subject = 130975

        data_info = self.data_dict[self.data_keys[idx]]
        #data_info = self.data_dict[130975]

        time_points = [key for key in data_info.keys()]
        time_point = time_points[random.randint(0, len(time_points)-1)]
        nodules = data_info[time_point]
        nifti = nib.load(data_info[time_point]['nifti_path'])
        binary_mask = np.load(os.path.join(self.root_masks, str(subject), f'T{time_point}', str(data_info[time_point]['nifti_path']).split('/')[-1].replace('_0000.nii.gz', '$mask.npy')))
        binary_mask = binary_mask.transpose(1,2,0)
        image = nifti.get_fdata()
        with open(data_info[time_point]['prediction_path'], 'rb') as pickle_file:
            prediction = pickle.load(pickle_file)
        filtered_predictions = self.extract_top_predictions_with_mask(prediction, binary_mask, top_x=5)

        center_coordinates = self.compute_bounding_box_centers(filtered_predictions)
        #nodule_arrays = []
        standardised_coordinates = self.compute_new_bounding_boxes(center_coordinates, self.crop_size)

        nodule_arrays = []

        for coordinates in standardised_coordinates:

            nodule = self.crop_3d_array(image, coordinates)
            if self.volumetric:
                #nodule = np.transpose(nodule, (2, 0, 1))
                nodule = np.expand_dims(nodule, axis=0)

            if self.transform:
                nodule = self.transform(nodule)

            #nodule = TioNormalizeStandard(self.mean, self.std_dev)(nodule)

            #nodule = torch.tensor(nodule)
               
            nodule = torch.cat([nodule, nodule, nodule], dim=0)
            nodule_arrays.append(nodule)

        

        stacked_nodules = torch.stack(nodule_arrays, axis=0)

        return stacked_nodules, nodules['label'], subject, time_point


    def extract_top_predictions(self, prediction_data, top_x=5):
        # Sort the indices of pred_scores in descending order
        top_indices = prediction_data['pred_scores'].argsort()[::-1][:top_x]

        # Extract the top predictions
        top_pred_boxes = prediction_data['pred_boxes'][top_indices]
        top_pred_scores = prediction_data['pred_scores'][top_indices]
        top_pred_labels = prediction_data['pred_labels'][top_indices]

        # Create a new dictionary with the same structure, but only with top predictions
        top_predictions = {
            'pred_boxes': top_pred_boxes,
            'pred_scores': top_pred_scores,
            'pred_labels': top_pred_labels,
            'restore': prediction_data['restore'],
            'original_size_of_raw_data': prediction_data['original_size_of_raw_data'],
            'itk_origin': prediction_data['itk_origin'],
            'itk_spacing': prediction_data['itk_spacing'],
            'itk_direction': prediction_data['itk_direction']
        }

        return top_predictions

    def extract_top_predictions_with_mask(self, prediction_data, binary_mask, top_x=5):
        # Sort the indices of pred_scores in descending order
        top_indices = prediction_data['pred_scores'].argsort()[::-1][:top_x]

        # Initialize lists to store the top predictions that are within the mask
        filtered_top_pred_boxes = []
        filtered_top_pred_scores = []
        filtered_top_pred_labels = []

        for index in top_indices:
            # Extract the bounding box
            box = prediction_data['pred_boxes'][index]

            # Calculate the center of the bounding box
            center_x = (box[4] + box[5]) / 2
            center_y = (box[1] + box[3]) / 2
            center_z = (box[0] + box[2]) / 2


            # Check if the center of the box is within the mask
            if binary_mask[int(center_x), int(center_y), int(center_z)]:
                # If center is in the mask, add the prediction to the filtered lists
                filtered_top_pred_boxes.append(box)
                filtered_top_pred_scores.append(prediction_data['pred_scores'][index])
                filtered_top_pred_labels.append(prediction_data['pred_labels'][index])

        # Check if the number of boxes is less than top_x
        while len(filtered_top_pred_boxes) < top_x:
            # Add an artificial box with all coordinates as 0
            filtered_top_pred_boxes.append([0,0,0,0,0,0])
            filtered_top_pred_scores.append(0)  # Assign a score of 0 for artificial boxes
            filtered_top_pred_labels.append(0)  # Label for the artificial box

        # Create a new dictionary with the same structure, but only with top predictions within the mask
        top_predictions = {
            'pred_boxes': np.array(filtered_top_pred_boxes),
            'pred_scores': np.array(filtered_top_pred_scores),
            'pred_labels': np.array(filtered_top_pred_labels),
            'restore': prediction_data['restore'],
            'original_size_of_raw_data': prediction_data['original_size_of_raw_data'],
            'itk_origin': prediction_data['itk_origin'],
            'itk_spacing': prediction_data['itk_spacing'],
            'itk_direction': prediction_data['itk_direction']
        }

        return top_predictions


    def compute_bounding_box_centers(self, filtered_predictions):
        centers = []

        for box in filtered_predictions['pred_boxes']:

            if np.all(box == 0):
                centers.append([0,0,0])
            else:
                # Calculate the midpoint for x, y, z
                z_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                x_center = (box[4] + box[5]) / 2

                centers.append([x_center, y_center, z_center])

        return np.array(centers)



    def get_crop_coordinates(self, center_coordinates):

        #returns the computed bounding box according to crop size
        #return [i, ((coordinate-self.crop_size[i], coordinate+self.crop_size[i])) for coordinate in enumerate(center_coordinates)]
        return [
                center_coordinates[0]-(self.crop_size[0]/2), center_coordinates[0]+(self.crop_size[0]/2), 
                center_coordinates[1]-(self.crop_size[1]/2), center_coordinates[1]+(self.crop_size[1]/2),
                center_coordinates[2]-(self.crop_size[2]/2), center_coordinates[2]+(self.crop_size[2]/2)
                ]


    def compute_new_bounding_boxes(self, centers, crop_size):
        new_bounding_boxes = []
        for center in centers:
            if np.all(center == 0):
                new_bounding_boxes.append([0,0,0,0,0,0])

            else:
                new_bbox = self.get_crop_coordinates(center)
                new_bounding_boxes.append(new_bbox)
        return new_bounding_boxes


    def crop_3d_array(self, array, bbox):

        # Ensure bounding box coordinates are within the array dimensions
        x1, x2 = int(np.floor(max(0, bbox[0]))), int(np.ceil(min(array.shape[0], bbox[1])))
        y1, y2 = int(np.floor(max(0, bbox[2]))), int(np.ceil(min(array.shape[1], bbox[3])))
        z1, z2 = int(np.floor(max(0, bbox[4]))), int(np.ceil(min(array.shape[2], bbox[5])))

            # Crop the array
        return array[x1:x2, y1:y2, z1:z2]



    def adjust_bboxes_for_resampling(self, bounding_boxes, original_spacing, new_spacing):
        # Calculate the scaling factors for each dimension
        scaling_factors = np.array(original_spacing) / np.array(new_spacing)

        # Adjust the bounding box coordinates
        # Only z-axis coordinates (indices 4 and 5) need to be scaled in this case
        resampled_boxes = bounding_boxes * scaling_factors

        return resampled_boxes






path = '/home/brandtj/Documents/projects/iderha/nlst/preprocessing/documents/datasets_pickle/for_testing_online_preprocessing.pkl'



transform = t.Compose([
                t.ScaleIntensityRange(a_min=-1400.0, a_max=200.0, b_min=0, b_max=1, clip=True),
                t.ResizeWithPadOrCrop(spatial_size=[64,64,32], method='symmetric', mode='constant')
                ])

dataset = Nodule_classifier_online_extraction(pickle_path=path, volumetric=True, windowed=True, transform=transform, crop_size=(64,64,32))
loader = DataLoader(batch_size=12, dataset=dataset, prefetch_factor=8, num_workers=32, shuffle=False)
nodules, labels, subject, timepoint = next(iter(loader))
print('stop here!')

#%%
