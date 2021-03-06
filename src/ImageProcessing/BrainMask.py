import numpy as np
import skimage.morphology as sk
from skimage.morphology import convex_hull_image

from src.Auxiliar.auxfun import trackcalls


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
        print("segmentation --> the shape is ={}".format(self.original.shape))

        # plt.imshow(self.original[252, :, :])
        # plt.show()
        bw_mask2 = self.threshold(input_array=self.original, threshold_value=1150)
        # plt.imshow(bw_mask2[252, :, :])
        mf_grad2 = sk.remove_small_objects(ar=bw_mask2, min_size=10000)
        # plt.show()
        # Dilation
        dilation_volume_bw = self.volume_dilation(vol=mf_grad2, size_of_str_elem=3)
        # plt.imshow(dilation_volume_bw[252, :, :])
        # plt.show()
        # Open
        opening_volume_bw = self.volume_opening(vol=dilation_volume_bw, size_of_str_elem=2)
        # plt.imshow(opening_volume_bw[252, :, :])
        # plt.show()
        # Dilation
        dilation_volume_bw2 = self.volume_dilation(vol=opening_volume_bw, size_of_str_elem=3)

        # plt.imshow(dilation_volume_bw2[252, :, :])
        # plt.show()

        convex_hull_erroded = self.covex_hull_mask(dilation_volume_bw2)
        final_mask = self.give_padding(convex_hull_erroded)
        return final_mask.astype(int)

    def covex_hull_mask(self, binary_3d_array):
        final_hull_mask = np.empty(self.original.shape)
        for ind in range(binary_3d_array.shape[2]):
            if True in binary_3d_array[:, :, ind]:
                chi = convex_hull_image(binary_3d_array[:, :, ind] == 1)
                final_hull_mask[:, :, ind] = self.image_erode(image=chi, size_of_str_elem=10)

            else:
                final_hull_mask[:, :, ind] = np.zeros(binary_3d_array[:, :, ind].shape)
        return final_hull_mask

    @staticmethod
    def threshold(input_array, threshold_value):
        one = np.ones(input_array.shape)
        zero = np.zeros(input_array.shape)
        return np.where(input_array > threshold_value, one, zero).astype(dtype=bool)

    @staticmethod
    def volume_dilation(vol, size_of_str_elem):
        str_elem = sk.ball(size_of_str_elem)
        return sk.binary_dilation(image=vol, selem=str_elem)

    @staticmethod
    def volume_opening(vol, size_of_str_elem):
        elem = sk.ball(size_of_str_elem)
        return sk.binary_opening(image=vol, selem=elem)

    @staticmethod
    def image_erode(image, size_of_str_elem):
        elem = sk.disk(size_of_str_elem)
        return sk.binary_erosion(image=image, selem=elem)

    @staticmethod
    def give_padding(array):
        array[1, :, :] = 0
        array[-1, :, :] = 0
        array[:, 1, :] = 0
        array[:, -1, :] = 0
        array[:, :, 1] = 0
        array[:, :, -1] = 0
        return array
