import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class RadarDataset(Dataset):
    def __init__(self, file_path, path_len, CWdata=True):
        self.file_path = file_path
        self.path_len = path_len
        self.CWdata = CWdata

        self.images, self.labels, self.noised, self.ForN = self.load_file2Spectrogam(self.file_path, self.path_len)

    def addNoise(self, feature, SNR): ## add piper noise
        feature_n = feature.copy()
        c, h, w = feature_n.shape
        mask = np.random.choice((0,1,2),size=(c,h,w),p=[SNR,(1-SNR)/2,(1-SNR)/2])
        feature_n[mask==1] = 0
        feature_n[mask==0] = 0
        feature_n[mask==2] = 0
        return feature_n

    def addGauseNoise(self, feature, SNR, mean=0.0, var=1.0):
        noise = np.random.normal(mean, var, feature.shape)
        signal_power = 1/(feature.shape[1]*feature.shape[2])*np.sum(np.power(feature, 2))
        noise_power = signal_power/np.power(10,(SNR/10))
        noise = (np.sqrt(noise_power)/np.std(noise))*noise
        feature_N = feature+noise
        feature_N = 1*(feature_N-np.min(feature_N)) / (np.max(feature_N)-np.min(feature_N))
        return feature_N, noise

    def addGauseNoise_simple(self, feature, SNR=0, mean=0, var=0.001, is_Tensor=False):
        if is_Tensor:
            feature = feature.detach().numpy()
        feature_n = feature.copy()
        shape = feature_n.shape
        g_noise = np.random.normal(loc=mean, scale=var, size=shape)
        feature_n += g_noise
        feature_n = 1 * (feature_n - np.min(feature_n)) / (np.max(feature_n) - np.min(feature_n))
        if is_Tensor:
            return torch.Tensor(feature_n)
        else:
            return feature_n

    def load_file2Spectrogam(self, file_path, path_len=0):
        wave_ForN = []
        wave_feature = []
        wave_noised = []
        labels = []
        labsIndex = []
        for file_path, sub_dirs, filenames in os.walk(file_path):
            if filenames:
                for filename in filenames:
                    feature = np.load(os.path.join(file_path, filename),allow_pickle=True)
                    if self.CWdata:
                        feature = feature.reshape(1, 128, 128) 
                    feature = 1*(feature-np.min(feature)) / (np.max(feature)-np.min(feature)) ## 1                
                    feature_noised = self.addGauseNoise_simple(feature)
                    wave_feature.append(feature)
                    wave_noised.append(feature_noised)
                    if file_path.split('/')[-1][:4] == 'fall':
                        labels.append(1)
                        wave_ForN.append(feature) ### keep the feature map for a fall image
                        continue
                    labels.append(0)
                    wave_ForN.append(np.zeros_like(feature))

        #shuffle the numpy, then convert into tensor
        wave_feature = np.array(wave_feature)
        labels = np.array(labels)
        wave_ForN = np.array(wave_ForN)
        wave_noised = np.array(wave_noised)
        ind_test = np.arange(len(labels))
        np.random.shuffle(ind_test)
        wave_feature, labels, wave_noised, wave_ForN = wave_feature[ind_test], labels[ind_test], wave_noised[ind_test], wave_ForN[ind_test]
        wave_feature = torch.Tensor(wave_feature)
        wave_noised = torch.Tensor(wave_noised)
        wave_ForN = torch.Tensor(wave_ForN)
        labels = torch.Tensor(labels)#.long()  #

        return wave_feature, labels, wave_noised, wave_ForN

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.noised[idx], self.ForN[idx]

if __name__ == '__main__':
    file_path = './Data/...'
    dataset = RadarDataset(file_path, len(file_path), CWdata=True)
    print(dataset.ForN.shape)
    plt.imshow(dataset.noised[2,0,:,:],cmap='jet')
    plt.show()
    plt.imshow(dataset.ForN[2,0,:,:],cmap='jet')
    plt.show()
