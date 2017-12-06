import skimage.morphology as sk
import numpy as np
from skimage.morphology import convex_hull_image
from src.auxfun import trackcalls


class BrainMask(object):
    def __init__(self, neuro_img_array):
        self.original = neuro_img_array
        self.brain_mask = self.segmentation()

    def get_brain_mask(self):
        if not self.segmentation.has_been_called:
            self.segmentation()
        return self.brain_mask

    def get_brain_mask_slice(self, z):
        return self.brain_mask[:, :, z]

    @trackcalls
    def segmentation(self):
        bw_mask2 = self.threshold(input_array=self.original, threshold_value=1000)
        mf_grad2 = sk.remove_small_objects(ar=bw_mask2, min_size=10000)

        # Dilation
        dilation_volume_bw = self.volume_dilation(vol=mf_grad2, size_of_str_elem=3)

        # Open
        opening_volume_bw = self.volume_opening(vol=dilation_volume_bw, size_of_str_elem=2)

        # Dilation
        dilation_volume_bw2 = self.volume_dilation(vol=opening_volume_bw, size_of_str_elem=3)

        final_mask = self.give_padding(self.covex_hull_mask(dilation_volume_bw2))
        return final_mask.astype(int)

    def covex_hull_mask(self, binary_3Darray):
        final_hull_mask = np.empty(self.original.shape)
        for ind in range(binary_3Darray.shape[2]):
            if True in binary_3Darray[:, :, ind]:
                final_hull_mask[:, :, ind] = convex_hull_image(binary_3Darray[:, :, ind] == 1)
            else:
                final_hull_mask[:, :, ind]= np.zeros(binary_3Darray[:, :, ind].shape)
        return final_hull_mask

    def threshold(self, input_array, threshold_value):
        one = np.ones(input_array.shape)
        zero = np.zeros(input_array.shape)
        return np.where(input_array > threshold_value, one, zero).astype(dtype=bool)

    def volume_dilation(self, vol, size_of_str_elem):
        str_elem = sk.ball(size_of_str_elem)
        return sk.binary_dilation(image=vol, selem=str_elem)

    def volume_opening(self, vol, size_of_str_elem):
        elem = sk.ball(size_of_str_elem)
        return sk.binary_opening(image=vol, selem=elem)

    def give_padding(self, array):
        array[1, :, :] = 0
        array[-1, :, :] = 0
        array[:, 1, :] = 0
        array[:, -1, :] = 0
        array[:, :, 1] = 0
        array[:, :, -1] = 0
        return array
