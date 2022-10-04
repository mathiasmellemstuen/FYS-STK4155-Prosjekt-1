from enum import Enum
from imageio import imread
import numpy as np
from FrankeFunction import FrankeFunctionNoised

class DataSamplesType(Enum): 
    REAL = 0
    TEST = 1

def create_data_samples_with_real_data(path, n):
    print(path)
    terrain = imread(path)
    terrain = terrain[:n, :n]

    x = np.linspace(0, 1, np.shape(terrain)[0])
    y = np.linspace(0, 1, np.shape(terrain)[1])
    x, y = np.meshgrid(x, y)
    z = terrain

    return x, y, z

def create_data_samples_with_franke(max_noise = 0.01): 

    x = np.arange(0, 1, 0.01)
    y = np.arange(0, 1, 0.01)
    x, y = np.meshgrid(x,y)
    z = FrankeFunctionNoised(x,y,max_noise)

    return x, y, z

def create_data_samples(data_samples_type: DataSamplesType, real_data_path="data/SRTM_data_Norway_1.tif", real_data_n = 1000, test_data_noise = 0.01): 

    if data_samples_type == DataSamplesType.REAL:
        return create_data_samples_with_real_data(real_data_path, real_data_n)
    elif data_samples_type == DataSamplesType.TEST:
        return create_data_samples_with_franke(test_data_noise)

    raise Exception("Invalid DataSamplesType")