import os
import torch
import pathlib
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import piq
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--fold1', type=str, default='', help='Fold1')
parser.add_argument('--fold2', type=str, default='', help='Fold2')
args = parser.parse_args()


path1 = pathlib.Path(args.fold1)
path2 = pathlib.Path(args.fold2)
files1 = list(path1.glob('*.png')) + list(path1.glob('*.jpg'))
files2 = list(path2.glob('*.png')) + list(path2.glob('*.jpg'))

files_name1 = []
files_name2 = []
for f in files1:
    files_name1.append(os.path.basename(f))
for f in files2:
    files_name2.append(os.path.basename(f))


L1 = 0
L2 = 0
ssim = 0
msssim = 0
psnr = 0
lpips = 0

n = 0
trans = transforms.ToTensor()

for i, img_name in enumerate(files_name1):

    if not img_name in files_name2:
        print('Not found!')
        continue

    j = files_name2.index(img_name)

    img1 = Image.open(files1[i])
    img1 = img1.convert('L')
    img1 = trans(img1).unsqueeze_(0)

    img2 = Image.open(files2[j])
    img2 = img2.convert('L')
    img2 = trans(img2).unsqueeze_(0)

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()

    L1 += nn.L1Loss()(img1, img2)
    L2 += nn.MSELoss()(img1, img2)
    ssim += piq.ssim(img1, img2, data_range=1.)
    msssim += piq.multi_scale_ssim(img1, img2, data_range=1.)
    psnr += piq.psnr(img1, img2, data_range=1., reduction='none')
    lpips += piq.LPIPS(reduction='none')(img1, img2)
    n += 1

    print('\rProcessing Image Metrics: {}/{} ({:.2f})'.format(i+1, len(files_name1), 100.*(i+1)/len(files_name1)), end='')


print('......Done!')
if n == 0:
    print('No files!')
else:
    print("L1: {}".format(L1 / n))
    print("L2: {}".format(L2 / n))
    print("SSIM: {}".format(ssim / n))
    print("MS-SSIM: {}".format(msssim / n))
    print('PSNR: {}'.format(psnr / n))
    print("LPIPS: {}".format(lpips / n))


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


class Images(Dataset):
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.files = list(self.path.glob('*.png'))

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        # img = img.convert('L')
        img = transforms.ToTensor()(img)
        return {'images': img}

    def __len__(self):
        return len(self.files)

x_features = piq.FID().compute_feats(DataLoader(Images(args.fold1), batch_size=64))
y_features = piq.FID().compute_feats(DataLoader(Images(args.fold2), batch_size=64))
fid = piq.FID()(x_features, y_features)


class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

class Images2(Dataset):
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.files = list(self.path.glob('*.png'))

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        # img = img.convert('L')
        img = transforms.ToTensor()(img)
        return [img]

    def __len__(self):
        return len(self.files)

is1 = inception_score(IgnoreLabelDataset(Images2(args.fold1)), cuda=True, batch_size=64, resize=True, splits=10)
is2 = inception_score(IgnoreLabelDataset(Images2(args.fold2)), cuda=True, batch_size=64, resize=True, splits=10)
isd = torch.dist(torch.FloatTensor(is1), torch.FloatTensor(is2), 1)


print("IS1: {}\tIS2: {}\tISD: {}".format(is1, is2, isd))
print("FID: {}".format(fid))
