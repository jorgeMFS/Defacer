import os
import sys

import nibabel as nib
import nrrd
import pydicom


class FileReader(object):
    def __init__(self, file_name_path):
        self.file_path, self.file_name = self.path_file(file_name_path)
        self.array = self.determine_format()

    @staticmethod
    def path_file(file_and_path):
        path, filename = os.path.split(file_and_path)
        return path, filename

    def determine_format(self):
        _, file_extension = os.path.splitext(self.file_name)
        if file_extension == '.nrrd':
            return self.nrrd_reader()
        elif file_extension == ".dcm":
            return self.dcm_reader()
        elif file_extension == ".nii":
            return self.nibabel_reader()
        else:
            print("not supported file, please try dcm, nifti or nrrd file formats")
            sys.exit(0)

    def nibabel_reader(self):
        os.chdir(self.file_path)
        return nib.load(self.file_name)

    def nrrd_reader(self):
        os.chdir(self.file_path)
        neuro_image_array, _ = nrrd.read(self.file_name)
        return neuro_image_array

    def dcm_reader(self):
        os.chdir(self.file_path)
        ds = pydicom.read_file(self.file_name).pixel_array
        return ds

    def get_array(self):
        return self.array
