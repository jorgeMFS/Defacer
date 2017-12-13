import argparse
import logging
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from src.ImageProcessing.NeuroImage import NeuroImage
from src.Viewer.CoronalView import coronal_view
from src.Viewer.SagitalView import sagital_view
from src.WRFiles.FileWriter import FileWriter
import os

def edge_mask(mask):
    """ Find the edges of a mask or masked image

    Parameters
    ----------
    mask : 3D array
        Binary mask (or masked image) with axis orientation LPS or RPS, and the
        non-brain region set to 0

    Returns
    -------
    2D array
        Outline of sagittal profile (PS orientation) of mask
    """

    # Sagittal profile
    brain = mask.any(axis=0)

    # Simple edge detection
    edgemask = 4 * brain - np.roll(brain, 1, 0) - np.roll(brain, -1, 0) - \
               np.roll(brain, 1, 1) - np.roll(brain, -1, 1) != 0
    return edgemask.astype('uint8')


def convex_hull(brain):
    """ Find the lower half of the convex hull of non-zero points

    Implements Andrew's monotone chain algorithm [0].

    [0] https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain

    Parameters
    ----------
    brain : 2D array
        2D array in PS axis ordering

    Returns
    -------
    (2, N) array
        Sequence of points in the lower half of the convex hull of brain
    """
    # convert brain to a list of points in an n x 2 matrix where n_i = (x,y)
    pts = np.vstack(np.nonzero(brain)).T

    def cross(o, a, b):
        return np.cross(a - o, b - o)

    # x_list = []
    # y_list = []
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    #     x = p[0]
    #     y = p[1]
    #
    #     # x_list.append(x)
    #     # y_list.append(y)
    # # print("x_list length= {}".format(x_list.__len__()))
    # # print("y_list length= {}".format(y_list.__len__()))
    #
    # x_list = np.array(lower).T[0]
    # y_list = np.array(lower).T[1]
    # # print(x_list)
    # fig, ax = plt.subplots()
    # plt.imshow(brain)
    # ax.invert_yaxis()
    # ax.xaxis.tick_top()
    # plt.plot(y_list,x_list, 'o')
    # plt.show()
    return np.array(lower).T


def flip_axes(data, perms, flips):
    """ Flip a data array along specified axes

    Parameters
    ----------
    data : 3D array
    perms : (3,) sequence of ints
        Axis permutations to perform
    flips : (3,) sequence of bools
        Sequence of indicators for whether to flip along each axis

    Returns
    -------
    3D array
    """
    data = np.transpose(data, perms)
    for axis in np.nonzero(flips)[0]:
        data = nib.orientations.flip_axis(data, axis)
    return data


def orient_xps(img, hemi='R'):
    """ Set image orientation to RPS or LPS

    Parameters
    ----------
    img : SpatialImage
        Nibabel image to be reoriented
    hemi : 'R' or 'L'
        Orientation of first axis of output image (default: 'R')

    Returns
    -------
    data : 3D array_like
        Re-oriented data array
    perm : (3,) sequence of ints
        Permutation of axes, relative to RAS
    flips : (3,) sequence of bools
        Sequence of indicators of axes flipped
    """
    axes = nib.orientations.aff2axcodes(img.affine)
    perm = ['RASLPI'.index(axis) % 3 for axis in axes]
    inv_perm = np.argsort(perm)
    # Flips are in RPS order
    flips = np.array(axes)[inv_perm] != np.array((hemi, 'P', 'S'))
    # We permute axes then flip, so inverse flips are also permuted
    return flip_axes(img.get_data(), inv_perm, flips), perm, flips[perm]


def quickshear(original, mask, buff=10, observe=False):
    """ Deface image using Quickshear algorithm

    Parameters
    ----------
    original : SpatialImage
        Nibabel image of anatomical scan, to be defaced
    mask : SpatialImage
        Nibabel image of skull-stripped brain mask or masked anatomical
    buff : int
        Distance from mask to set shearing plane
    observe: bool
        Observe or not plots
    Returns
    -------
    SpatialImage
        Nibabel image of defaced anatomical scan
    """

    anat, anat_perm, anat_flip = orient_xps(original)
    mask, mask_perm, mask_flip = orient_xps(mask)

    edgemask = edge_mask(mask)
    low = convex_hull(edgemask)

    x_min = np.min(low[1])
    index = np.where(low[1] == x_min)
    index = index[0][0]
    y_min = low[0][index]
    y0 = low[0][0]
    x0 = low[1][0]
    slope = (y_min - y0) / (x_min - x0 - buff)
    # b = y - mx
    yint = low[1][0] - (low[0][0] * slope)
    ys = np.arange(0, mask.shape[1]) * slope + yint
    defaced_mask = np.ones(mask.shape, dtype='bool')

    for x, y in zip(np.nonzero(ys > 0)[0], ys.astype(int)):
        defaced_mask[:, x, :y] = 0

    print("defaced_mask shape  = {}".format(defaced_mask.shape))

    if observe:
        fig, ax = plt.subplots()
        plt.imshow(anat[256, :, :], cmap='gray')
        plt.imshow(mask[256, :, :], cmap='jet', alpha=0.5)
        plt.imshow(defaced_mask[256, :, :], cmap='jet', alpha=0.5)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        plt.plot(163, 0, 'o')
        plt.show()

    return original.__class__(
        flip_axes(defaced_mask * anat, anat_perm, anat_flip),
        original.affine, original.header)


def defacer(absolute_file_path_received):

    # Create NeuroImage Mask
    neuro_array = NeuroImage(absolute_path=absolute_file_path_received)
    img = neuro_array.get_ndarray()
    mask = neuro_array.get_neuromask()

    # Quickshear
    mask_nib = nib.Nifti1Image(mask, affine=np.eye(4))
    anat_nib = nib.Nifti1Image(img, affine=np.eye(4))
    quick = quickshear(anat_nib, mask_nib)
    defaced_array = quick.get_data()

    # Save
    _, filename = os.path.split(absolute_file_path_received)
    name, file_extension = os.path.splitext(filename)
    defaced_filename = name + "_defaced" + '.nii'
    save_file_name = os.path.join('/home/mikejpeg/IdeaProjects/Defacer/src/Webservices/tmp/', defaced_filename)
    FileWriter(file_name_path=save_file_name, np_array=defaced_array)
    return save_file_name


def main(observe=False):
    global anat_img
    global mask_img
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    filename_path = '/home/mikejpeg/IdeaProjects/Defacer/image/test_images/img2.nrrd'
    save_file_name = '/home/mikejpeg/IdeaProjects/Defacer/image/results/dicom/tester.dcm'

    parser = argparse.ArgumentParser(
        description='defacer for neuroimages',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('anat_file', type=str, nargs='?', default=filename_path,
                        help="full path with filename of neuroimage to deface")
    parser.add_argument('mask_file', type=str, nargs='?', default="",
                        help="full path with filename of brain mask")
    parser.add_argument('defaced_file', type=str, nargs='?', default=save_file_name,
                        help="full path with filename of defaced output image")

    opts = parser.parse_args()

    if opts.anat_file != filename_path:
        if opts.mask_file != "":
            anat_img = nib.load(opts.anat_file)
            mask_img = nib.load(opts.mask_file)

        else:
            neuro_array = NeuroImage(absolute_path=opts.anat_file)
            anat_img = neuro_array.get_ndarray()
            mask_img = neuro_array.get_neuromask()

    elif opts.anat_file == filename_path:

        # Creating NeuroImageObject and getting array.
        neuro_array = NeuroImage(absolute_path=opts.anat_file)
        anat_img = neuro_array.get_ndarray()
        mask_img = neuro_array.get_neuromask()

        if observe:  # Visualizing mask and volume
            sagital_view(mask_img)
            sagital_view(anat_img)

    # Changing axis to allow for correct defacing
    anat_img = np.swapaxes(anat_img, 1, 2)
    mask_img = np.swapaxes(mask_img, 1, 2)

    # Transforming into Nibabel file
    mask_img = nib.Nifti1Image(mask_img, affine=np.eye(4))
    anat_img = nib.Nifti1Image(anat_img, affine=np.eye(4))

    if anat_img.shape != mask_img.shape:
        logger.warning(
            "Anatomical and mask images do not have the same dimensions.")
        return -1

    new_anat = quickshear(anat_img, mask_img)
    new_anat.to_filename('/home/mikejpeg/IdeaProjects/Defacer/image/results/test.nii')
    new_anat = new_anat.get_data()
    if observe:
        sagital_view(new_anat)
        coronal_view(new_anat)

    if opts.defaced_file != save_file_name:
        FileWriter(file_name_path=opts.defaced_file, np_array=new_anat)
    else:
        FileWriter(file_name_path=save_file_name, np_array=new_anat)

    logger.info("Defaced file: {0}".format(opts.defaced_file))
    return 0


if __name__ == '__main__':
    sys.exit(main())
