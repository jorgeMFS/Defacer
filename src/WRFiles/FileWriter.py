import datetime
import os
import sys
import time

import nibabel as nib
import nrrd
import numpy as np
from pydicom.dataset import Dataset, FileDataset


class FileWriter(object):
    def __init__(self, file_name_path, np_array):
        self.save_file_path, self.save_file_name = self.path_file(file_name_path)
        self.np_array = np_array
        self.determine_format()

    @staticmethod
    def path_file(file_and_path):
        path, filename = os.path.split(file_and_path)
        return path, filename

    def determine_format(self):
        _, file_extension = os.path.splitext(self.save_file_name)
        if file_extension == '.nrrd':
            self.nrrd_writer()
        elif file_extension == ".dcm":
            self.dcm_writer()
        elif file_extension == ".nii":
            self.nibabel_writer()
        else:
            print("not supported file, please try dcm, nifti or nrrd file formats")
            sys.exit(0)

    def nrrd_writer(self):
        os.chdir(self.save_file_path)
        nrrd.write(self.save_file_name, self.np_array)

    def nibabel_writer(self):
        os.chdir(self.save_file_path)
        save_img = nib.Nifti1Image(self.np_array, affine=np.eye(4))
        nib.save(save_img, filename=self.save_file_name)

    def dcm_writer(self):
        if self.np_array.ndim == 2:
            os.chdir(self.save_file_path)
            self.write_dicom(self.np_array, self.save_file_name)
        elif self.np_array.ndim == 3:
            os.chdir(self.save_file_path)
            self.create_multiple_files()

    @staticmethod
    def write_dicom(pixel_array, dcm_name):
        """
        Input:
        pixel_array: 2D numpy ndarray.
        If pixel_array is larger than 2D, errors.
        filename: string name for the output file.
        """

        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = 'Secondary Capture Image Storage'
        file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
        file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.0.100.4.0'
        ds = FileDataset(dcm_name, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.Modality = 'WSD'
        ds.ContentDate = str(datetime.date.today()).replace('-', '')
        ds.ContentTime = str(time.time())  # milliseconds since the epoch
        ds.StudyInstanceUID = '1.3.6.1.4.1.9590.100.1.1.124313977412360175234271287472804872093'
        ds.SeriesInstanceUID = '1.3.6.1.4.1.9590.100.1.1.369231118011061003403421859172643143649'
        ds.SOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
        ds.SOPClassUID = 'Secondary Capture Image Storage'
        ds.SecondaryCaptureDeviceManufctur = 'Python 3'

        # These are the necessary imaging components of the FileDataset object.
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.HighBit = 15
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.SmallestImagePixelValue = b'\\x00\\x00'
        ds.LargestImagePixelValue = b'\\xff\\xff'
        ds.Columns = pixel_array.shape[0]
        ds.Rows = pixel_array.shape[1]
        if pixel_array.dtype != np.uint16:
            pixel_array = pixel_array.astype(np.uint16)
        ds.PixelData = pixel_array.tostring()

        ds.save_as(dcm_name)
        return

    def create_multiple_files(self):

        array = self.np_array
        _, string_file_name = os.path.splitext(self.save_file_name)
        index = np.where(array.shape == np.min(array.shape))[0]
        if index == 0:
            for it in range(np.min(array.shape)):
                px_array = array[it, :, :]
                ss = string_file_name + '_' + str(it) + '.dcm'
                self.write_dicom(pixel_array=px_array, dcm_name=ss)
        if index == 1:
            print(np.max(array.shape))
            for it in range(np.min(array.shape)):
                px_array = array[:, it, :]
                ss = string_file_name + '_' + str(it) + '.dcm'
                self.write_dicom(pixel_array=px_array, dcm_name=ss)
        if index == 2:
            for it in range(np.min(array.shape)):
                px_array = array[:, :, it]
                ss = string_file_name + '_' + str(it) + '.dcm'
                self.write_dicom(pixel_array=px_array, dcm_name=ss)


def main():
    x = np.arange(16).reshape(16, 1)
    pixel_array = (x + x.T) * 32
    pixel_array = np.tile(pixel_array, (16, 16))
    file_path = '/home/mikejpeg/IdeaProjects/Defacer/image/results/dicom/pretty.dcm'
    FileWriter(file_name_path=file_path, np_array=pixel_array)
    return 0


if __name__ == '__main__':
    sys.exit(main())
