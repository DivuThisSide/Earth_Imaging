# Libraries
import os, json, tarfile, warnings
warnings.filterwarnings('ignore')


import numpy as np
import requests
import rioxarray
import rasterio
from rasterio.mask    import mask as rio_mask
from rasterio.warp   import transform_geom, calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
from shapely.geometry import box, mapping, shape
from scipy.ndimage   import zoom
from datetime import datetime


import matplotlib.dates as mdates
import matplotlib.pyplot    as plt
import matplotlib.patches   as mpatches
import matplotlib.colors    as mcolors
from   matplotlib.colorbar  import ColorbarBase


from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import (confusion_matrix, classification_report,
                                     f1_score, roc_curve, auc,
                                     ConfusionMatrixDisplay)
import torch
import torch.nn              as nn
import torch.optim           as optim
from   torch.utils.data      import DataLoader, TensorDataset
