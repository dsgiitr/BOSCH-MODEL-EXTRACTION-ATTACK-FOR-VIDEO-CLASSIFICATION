# Imports for transform and dataset prepration
import numpy as np
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.utils.data import TensorDataset, DataLoader, Dataset, WeightedRandomSampler
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, top_k_accuracy_score
import torch.nn.functional as func
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch
from torch.autograd import Variable
import sys
from tqdm import tqdm

from models.VideoSwin import PretrainedVideoSwinTransformer
# from models.MoviNet import MoviNet
from models.ResNet3D import ResNet3D
from c3d_pytorch.C3D_model import C3D
from models.Generator import Generator
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import itertools
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#from movinets import MoViNet
#from movinets.config import _C
# from c3d_pytorch.C3D_model import C3D

def tofloat(x):
  return x[:32].float()

def collate_fn(batch):
    # print(batch[:10])
    x = torch.stack([torch.tensor(data_item[0]) for data_item in batch])
    y = [int(data_item[2]) for data_item in batch]
    # return x[:32], y
    return x, y

train_transform = transforms.Compose([
                    #transforms.Lambda(lambda x: x / 255.0),
                    #transforms.functional.uniform_temporal_subsample_repeated(32, (1,0), temporal_dim = 2),
                    tofloat,
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.Normalize((123, 116, 103), (58, 57, 57)),
                    # transforms.RandomHorizontalFlip(p=0.5),
                    # transforms.ColorJitter(brightness=25/225),
                    # transforms.RandomRotation(15)
                  ])

train_kinetics = datasets.Kinetics("../k400val_pytorch", frames_per_clip=32, split='val', num_classes= '400', step_between_clips=2000000, transform = train_transform, download=False, num_download_workers=1, num_workers=80)
test_ds = train_kinetics
test_dl = DataLoader(test_ds, collate_fn=collate_fn, batch_size=16, shuffle = True);

student = ResNet3D(pretrained=False, n_classes=400).to(device)
student = torch.load('student.pth')
teacher = PretrainedVideoSwinTransformer('checkpoints/swin_base_patch244_window877_kinetics400_1k.pth').to(device)

size_changer = torch.nn.AvgPool3d((1, 2, 2), stride=None, padding=0, ceil_mode=False)

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]

acc5 = []
acc1 = []
student.eval()
with torch.no_grad():
    for video, label in tqdm(test_dl):
        video = Variable(video.to(device), requires_grad=False)
        video = video.permute(0, 2, 1, 3, 4)
        l_ = teacher(video)
        l_ = torch.nn.functional.gumbel_softmax(l_, tau=1, hard=False, eps=1e-10, dim=- 1)
        video = size_changer(video)
        prediction = student(video)
        acc = accuracy(output=prediction.cpu(), target=torch.argmax(l_, dim=1).cpu(), topk=(1,5))
        acc1.append(acc[0])
        acc5.append(acc[1])
        print(torch.mean(torch.stack(acc5)), torch.mean(torch.stack(acc1)))
print(">>>>>>>>>>>>>>>>>>>>>>>>>", torch.mean(torch.stack(acc5)), torch.mean(torch.stack(acc1)))