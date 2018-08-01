import os

import nibabel as nib
import numpy as np
import skimage.morphology as sk

from src.Auxiliar.auxfun import trackcalls
from src.ImageProcessing.BrainMask import BrainMask
from src.Viewer.CoronalView import coronal_view
from src.Viewer.SagitalView import sagital_view
from src.WRFiles.FileReader import FileReader


class NeuroImage(object):
    def __init__(self, absolute_path):
        self.neuro_image_array = FileReader(absolute_path).get_array()
        self.brain_mask = self.create_brain_mask()

    def get_ndarray(self):
        return self.neuro_image_array

    def get_neuromask(self):
        if not self.create_brain_mask.has_been_called:
            self.create_brain_mask()
        return self.brain_mask

    @trackcalls
    def create_brain_mask(self):
        brain_mask = BrainMask(self.neuro_image_array)
        return brain_mask.get_brain_mask()

    def apply_mask(self,bw_volume, bw_volume_mask, size_of_filtered_particles):
        vascular_structure = np.multiply(bw_volume, bw_volume_mask)
        return sk.remove_small_objects(ar=vascular_structure, min_size=size_of_filtered_particles)


    def save_as_original_as_nifty(self, name,save_path='/home/mikejpeg/IdeaProjects/Defacer/image/results/'):
        os.chdir(save_path)
        save_img = nib.Nifti1Image(self.neuro_image_array, affine=np.eye(4))
        nib.save(save_img, filename=name)

    def save_mask_as_nifty(self, name, save_path='/home/mikejpeg/IdeaProjects/Defacer/image/results'):
        os.chdir(save_path)
        save_img = nib.Nifti1Image(restructure_array(self.brain_mask), affine=np.eye(4))
        nib.save(save_img, filename=name)


def restructure_array(array):
    out_array = np.reshape(array, (array.shape[0], array.shape[1], array.shape[2], 1, 1, 1))
    print(out_array.shape)
    return out_array


if __name__ == '__main__':
    filename_path = '/home/mikejpeg/IdeaProjects/Defacer/image/test_images/img4.nrrd'
    neuro_array = NeuroImage(absolute_path=filename_path)
    nd = neuro_array.get_ndarray()
    print(type(nd))
    print(nd.shape)

    neuro_array.create_brain_mask()
    brain_mask = neuro_array.get_neuromask()
    print(type(brain_mask))
    print(brain_mask.shape)
    sagital_view(brain_mask)
    coronal_view(brain_mask)
