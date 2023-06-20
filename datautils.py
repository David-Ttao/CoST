import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
from torch.utils.data import TensorDataset, DataLoader,ConcatDataset
import torch
import math
from cost import PretrainDataset
import random

def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')    
    if univar:
        data = data[: -1:]
        
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0

def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)

def load_forecast_csv(name, univar=False):
    data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]
    
    # if univar:
    #     data = data[['OT']]
        
        
    data = data.to_numpy()
    # if name == 'ETTh1' or name == 'ETTh2':
    #     train_slice = slice(None, 12*30*24)
    #     valid_slice = slice(12*30*24, 16*30*24)
    #     test_slice = slice(16*30*24, 20*30*24)
    # elif name == 'ETTm1' or name == 'ETTm2':
    #     train_slice = slice(None, 12*30*24*4)
    #     valid_slice = slice(12*30*24*4, 16*30*24*4)
    #     test_slice = slice(16*30*24*4, 20*30*24*4)
    # else:
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    # if name in ('electricity'):
    #     data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    # else:
    data = np.expand_dims(data, 0)
    
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)

    datatemp=np.concatenate([data[:,:,:dt_embed.shape[2]], data[:,:,dt_embed.shape[2]:dt_embed.shape[2]+1]], axis=-1)
    for i in range(dt_embed.shape[2]+1,data.shape[2]):
        temp_channel=np.concatenate([data[:,:,:dt_embed.shape[2]], data[:,:,i:i+1]], axis=-1)
        datatemp=np.concatenate([datatemp,temp_channel],axis=0)
    
   
    pred_lens = [96]
 
        
    return datatemp, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols



def load_muti_forecast_csv(args,name,univar=True):
    data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]
   
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)
    ##
   
    dt_scaler = StandardScaler().fit(dt_embed[train_slice])
    dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
    data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)
    ##
   
    ##可以更改
    pred_lens = [96]

    train_data=data[:,train_slice]
    sections = train_data.shape[1] // args.max_train_length
    if sections >= 2:
        train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)
    temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
    if temporal_missing[0] or temporal_missing[-1]:
        train_data = centerize_vary_length_series(train_data)
    train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
    multiplier = 1 if train_data.shape[0] >= args.batch_size else math.ceil(args.batch_size / train_data.shape[0])
    
    single_dset=[]
    for i in range(dt_embed.shape[2],train_data.shape[2]):
        temp=train_data[:,:,i:i+1]
        tempdata=np.concatenate([train_data[:,:,:dt_embed.shape[2]], train_data[:,:,i:i+1]], axis=-1)
        #train_dataset=TensorDataset(torch.from_numpy(tempdata).to(torch.float))
        train_dataset = PretrainDataset(torch.from_numpy(tempdata).to(torch.float), sigma=0.5, multiplier=multiplier)
        single_dset.append(train_dataset)
    Concat_single_dataset=ConcatDataset(single_dset)
    train_loader = DataLoader(Concat_single_dataset, batch_size=min(args.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
    return train_loader, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def set_device(usage=5, device=None):
    "set the device that has usage < default usage  "
    if device:
        torch.cuda.set_device(device)
        return
    device_ids = get_available_cuda(usage=usage)
    torch.cuda.set_device(device_ids[0])   # get the first available device
    
def get_available_cuda(usage=10):
    if not torch.cuda.is_available(): return
    # collect available cuda devices, only collect devices that has less that 'usage' percent 
    device_ids = []
    for device in range(torch.cuda.device_count()):
        if torch.cuda.utilization(device) < usage: device_ids.append(device)
    return device_ids        