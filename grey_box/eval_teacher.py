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

from movinets import MoViNet
from movinets.config import _C

if (len(sys.argv)!=2):
  print('Usage: python3 eval_teacher.py [MODEL_NAME]')
  exit(0)

# Train transform and other utils
DEVICE = 'cuda:0'
FRAMES = 32
BATCH_SIZE = 8
STEPS_BETWEEN_CLIPS = 20000000


def tofloat(x):
  return x[:FRAMES].float()
  # return x.float()

train_transform_swin = transforms.Compose([
                    #transforms.Lambda(lambda x: x / 255.0),
                    #transforms.functional.uniform_temporal_subsample_repeated(32, (1,0), temporal_dim = 2),
                    tofloat,
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.Normalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375)),
                    #transforms.RandomHorizontalFlip(p=0.5),
                    #transforms.ColorJitter(brightness=25/225),
                    #transforms.RandomRotation(15)
                  ])

train_transform_movinet = transforms.Compose([
                    transforms.Lambda(lambda x: x / 255.0),
                    #transforms.functional.uniform_temporal_subsample_repeated(32, (1,0), temporal_dim = 2),
                    tofloat,
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    # transforms.Normalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375)),
                    #transforms.RandomHorizontalFlip(p=0.5),
                    #transforms.ColorJitter(brightness=25/225),
                    #transforms.RandomRotation(15)
                  ])

def collate_fn(batch):
    x = torch.stack([torch.tensor(data_item[0]) for data_item in batch])
    x = x[:, ::2, :, :, :]
    y = [int(data_item[2]) for data_item in batch]
    # return x[:32], y
    return x, y

VALIDATION = '../k400val_pytorch'

train_kinetics_movinet = datasets.Kinetics(VALIDATION, frames_per_clip= FRAMES, split='val', num_classes= '400', step_between_clips= STEPS_BETWEEN_CLIPS, transform = train_transform_movinet,  download= False, num_download_workers= 1, num_workers= 80)
train_kinetics_swin = datasets.Kinetics(VALIDATION, frames_per_clip= FRAMES, split='val', num_classes= '400', step_between_clips= STEPS_BETWEEN_CLIPS, transform = train_transform_swin,  download= False, num_download_workers= 1, num_workers= 80)

#train_kinetics = datasets.Kinetics(VALIDATION, frames_per_clip= FRAMES, split='val', num_classes= '400', step_between_clips= STEPS_BETWEEN_CLIPS, transform = train_transform,  download= False, num_download_workers= 1, num_workers= 80)
#train_kinetics = datasets.Kinetics("../k400samples", frames_per_clip= FRAMES, split='val', num_classes= '400', step_between_clips= STEPS_BETWEEN_CLIPS, transform = train_transform,  download= False, num_download_workers= 1, num_workers= 80)
# train_ds = train_kinetics
# test_ds = train_kinetics
# train_dl = DataLoader(train_ds, collate_fn=collate_fn, batch_size = batch_size, shuffle = True);

test_dl_swin = DataLoader(train_kinetics_swin, collate_fn=collate_fn, batch_size = BATCH_SIZE, shuffle = True);
test_dl_movinet = DataLoader(train_kinetics_movinet, collate_fn=collate_fn, batch_size = BATCH_SIZE, shuffle = True);


# Main code for network extraction


torch.cuda.empty_cache()
import warnings
warnings.filterwarnings('ignore')
from ignite.metrics import Accuracy, Loss
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from mmaction.datasets import build_dataloader, build_dataset
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmaction.models import build_model
from mmcv import Config, DictAction

config = 'configs/recognition/swin/swin_base_patch244_window877_kinetics400_1k.py'
checkpoint = '../swin_base_patch244_window877_kinetics400_1k.pth'


class MMActionModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X):
        return self.model.forward_dummy(X)[0]

def topk(output, target, maxk=5):
    """Computes the precision@k for the specified value of maxk"""
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = correct[:maxk].view(-1).float().sum(0)
    return correct_k.mul_(100.0 / batch_size)

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.
    """
    with torch.no_grad():
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth

        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]

def train_with_extraction(victim, teacher):
    with torch.no_grad():
        acc5 = []
        acc1 = []
        if (teacher==1):
          test_dl = test_dl_movinet
        else:
          test_dl = test_dl_swin
        for step,(video, label) in tqdm(enumerate(test_dl)):
            #if step > 5:
            #    break
            video = Variable(video.to(DEVICE), requires_grad=False)
            video = video.permute(0, 2, 1, 3, 4)
            #print("videoshape >>>>", video.size())
            l_ = victim(video)
            l_ = torch.nn.functional.gumbel_softmax(l_, tau=1, hard=False, eps=1e-10, dim=- 1)

            print(label)
            print(torch.argmax(l_, dim=1).cpu())
            #acc1.append(top_k_accuracy_score(torch.argmax(l_, dim=1).cpu(), label, k=1))
            #acc5.append(top_k_accuracy_score(torch.argmax(l_, dim=1).cpu(), label, k=5))
            acc = accuracy(output=l_.cpu(), target=torch.tensor(label), topk=(1,5))
            acc1.append(acc[0])
            acc5.append(acc[1])
    #print(">>>>>>>>>>",acc5, acc1)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>", torch.mean(torch.stack(acc5)), torch.mean(torch.stack(acc1)))


cfg = Config.fromfile(config)
if __name__ == '__main__':
    teacher = sys.argv[1]
    if ('swin' in teacher):
      model_victim = build_model(cfg.model, train_cfg=None, test_cfg=None)
      load_checkpoint(model_victim, checkpoint, map_location=DEVICE)
      model_victim.to(DEVICE)
      victim = MMActionModelWrapper(model_victim)
      option = 0
    else:
      victim = MoViNet(_C.MODEL.MoViNetA2, causal = False, pretrained = True )
      victim.to(DEVICE)
      option = 1
    
    for param in victim.parameters():
      param.requires_grad = False
    victim.eval()

    # size_changer = torch.nn.AvgPool3d((1, 2, 2), stride=None, padding=0, ceil_mode=False)

    print(f'Evaluating {teacher} on k400val.\nConfiguartion: \nBATCH_SIZE={BATCH_SIZE}\nFRAMES{FRAMES}')
    train_with_extraction(victim, option)
