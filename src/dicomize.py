from dicom.dataset import Dataset, FileDataset
import numpy as np
import datetime, time
import os


def write_dicom(pixel_array,dcm_name):
    """
    INPUTS:
    pixel_array: 2D numpy ndarray.  If pixel_array is larger than 2D, errors.
    filename: string name for the output file.
    """

    ## This code block was taken from the output of a MATLAB secondary
    ## capture.  I do not know what the long dotted UIDs mean, but
    ## this code works.
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = 'Secondary Capture Image Storage'
    file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
    file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.0.100.4.0'
    ds = FileDataset(dcm_name, {},file_meta = file_meta,preamble=b"\0"*128)
    ds.Modality = 'WSD'
    ds.ContentDate = str(datetime.date.today()).replace('-','')
    ds.ContentTime = str(time.time()) #milliseconds since the epoch
    ds.StudyInstanceUID =  '1.3.6.1.4.1.9590.100.1.1.124313977412360175234271287472804872093'
    ds.SeriesInstanceUID = '1.3.6.1.4.1.9590.100.1.1.369231118011061003403421859172643143649'
    ds.SOPInstanceUID =    '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
    ds.SOPClassUID = 'Secondary Capture Image Storage'
    ds.SecondaryCaptureDeviceManufctur = 'Python 3'

    ## These are the necessary imaging components of the FileDataset object.
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


def create_multiple_files(array,string_file_name):
    index = np.where(array.shape == np.min(array.shape))[0]
    if index == 0:
        for it in range(np.max(array.shape)):
            px_array = array[it, :, :]
            ss = string_file_name + '_' + str(it)
            write_dicom(pixel_array=px_array,dcm_name=ss)
    if index == 1:
        print(np.max(array.shape))
        for it in range(np.max(array.shape)):
            px_array = array[:, it, :]
            ss = string_file_name + '_' + str(it)
            write_dicom(pixel_array=px_array, dcm_name=ss)
    if index == 2:
        for it in range(np.max(array.shape)):
            px_array = array[:, :, it]
            ss = string_file_name + '_' + str(it)
            write_dicom(pixel_array=px_array, dcm_name=ss)


def ch_directory(path):
    os.chdir(path)


if __name__ == "__main__":
    x = np.arange(16).reshape(16,1)
    pixel_array = (x + x.T) * 32
    pixel_array = np.tile(pixel_array,(16,16))
    write_dicom(pixel_array,'/home/mikejpeg/IdeaProjects/Defacer/image/results/dicom/pretty.dcm')