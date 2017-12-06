import dicom
import nibabel as nib
import numpy as np
from PIL import Image
import functools


def trackcalls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.has_been_called = True
        return func(*args, **kwargs)
    wrapper.has_been_called = False
    return wrapper



def get_dicom_pixel_data(dicom_directory):
    ds = dicom.read_file(dicom_directory).pixel_array
    return ds


def save_as_nifti(numpy_array, name):
    save_img = nib.Nifti1Image(numpy_array, affine=np.eye(4))
    nib.save(save_img, filename=name)


def read_nifti(path):
    img = nib.load(filename=path)
    image_data = img.get_data()
    return image_data


def transform_array_img(data):
    img = Image.fromarray(data)
    return img


def restructure_array(array):
    out_array = np.reshape(array, (array.shape[0], array.shape[1], array.shape[2], 1, 1, 1))
    print(out_array.shape)
    return out_array


def broadcast(input_matrix):
    return (((input_matrix - input_matrix.min()) / (input_matrix.max() - input_matrix.min())) * 4052).astype(np.uint16)


def houndsfield(input_matrix, slope=1, intercept=0):
    return input_matrix * slope + intercept


if __name__ == '__main__':
    ''' read file '''

    pth_nifti_file = "C:\\Users\\Bioinformatics-IEETA\\Documents\\nifti\\registered_CT.nii"
    ct_exam = read_nifti(pth_nifti_file)
    ct_exam = broadcast(ct_exam)
    print(ct_exam.shape)
    ct_exam = restructure_array(ct_exam)
    print(ct_exam.shape)
    filename = "C:\\Users\\Bioinformatics-IEETA\\Documents\\nifti\\registered_CT_range.nii"
    save_as_nifti(ct_exam, filename)
    # print(ct_exam[:, :, 1, 1])
    # print('shape', ct_exam.shape)
    # print('min', np.min(ct_exam))
    # print('max', np.max(ct_exam))
