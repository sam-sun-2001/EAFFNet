import numpy as np
from osgeo import gdal # 遥感图像读取当然是用gdal
def readimage(path):
    dataset=gdal.Open(path)
    bandnum=dataset.RasterCount
    x=dataset.RasterXSize
    y=dataset.RasterYSize
    data=np.array(dataset.ReadAsArray(0,0,x,y))
    return data
