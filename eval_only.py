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


if (len(sys.argv)!=3):
    print('Usage : python3 eval.py [C3D, r21] weight_PATH')
    exit(0)
# Train transform and other utils
BATCH_SIZE = 16
FRAME = 32

def tofloat(x):
  return x[:FRAME].float()
  # return x.float()


train_transform = transforms.Compose([
                    transforms.Lambda(lambda x: x / 255.0),
                    #transforms.functional.uniform_temporal_subsample_repeated(32, (1,0), temporal_dim = 2),
                    tofloat,
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    # transforms.Normalize((123, 116, 103), (58, 57, 57)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=25/225),
                    transforms.RandomRotation(15)
                  ])


test_transform = transforms.Compose([
                    transforms.Lambda(lambda x: x / 255.0),
                    #transforms.functional.uniform_temporal_subsample_repeated(32, (1,0), temporal_dim = 2),
                    tofloat,
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    # transforms.Normalize((123, 116, 103), (58, 57, 57)),
                    #transforms.RandomHorizontalFlip(p=0.5),
                    #transforms.ColorJitter(brightness=25/225),
                    #transforms.RandomRotation(15)
                  ])


# Load Dataset

def collate_fn(batch):
    # print(batch[:10])
    x = torch.stack([torch.tensor(data_item[0]) for data_item in batch])
    y = [int(data_item[2]) for data_item in batch]
    y = torch.tensor(y)
    # return x[:32], y
    return x, y

VALIDATION_UCF = 'ucf101'
VALIDATION_HMDB = 'hmdb51'

test_ucf = datasets.Kinetics(VALIDATION_HMDB, split='train', frames_per_clip= FRAME, step_between_clips = 32, transform = test_transform, download=False, num_workers= 80)
test_hmdb51 = datasets.Kinetics(VALIDATION_UCF, split='train', frames_per_clip= FRAME, step_between_clips = 32, transform = test_transform, download=False, num_workers= 80)
test_ds = torch.utils.data.ConcatDataset([test_ucf, test_hmdb51])

test_dl = DataLoader(test_ds, collate_fn=collate_fn, batch_size = batch_size, shuffle = True);

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

DEVICE = 'cuda:0'
SPATIAL_DIM = 224
TEMPORAL_DIM = 16
NUM_CHANNELS = 3

class MMActionModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X):
        return self.model.forward_dummy(X)[0]

def size_changer(x, tm, sz):
      return torch.nn.functional.upsample(x, size=(tm,sz,sz), scale_factor=None, mode='nearest', align_corners=None)

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

def evaluate(model):
    ''' Given the model and saved_weights evaluate it with top5k'''
    model.eval()
    with torch.no_grad():
        acc1 = []
        acc5 = []
        e = tqdm(test_dl)
        for step,(video, label) in enumerate(e):
            #if step > 3:
            #    break
            video = Variable(video.to(DEVICE), requires_grad=False)
            video = video.permute(0, 2, 1, 3, 4)
            video = size_changer(video, FRAME, 112)
            prediction = model(video)

            # print(f'Predicted class: {torch.argmax(prediction, dim=1)}, Teacher class: {torch.argmax(l_, dim=1)}, Actual label: {label}')

            acc_list = accuracy(prediction.cpu(), label.cpu(), topk=(1,5))
            acc1.append(acc_list[0])
            acc5.append(acc_list[1])
    print('top1 =',torch.mean(torch.stack(acc1)))
    print('top5 =',torch.mean(torch.stack(acc5)))


if __name__ == '__main__':
    print('Usage : python3 eval.py [C3D, r21] weight_PATH')
    model_choice = sys.argv[1]
    adversary = torch.load(sys.argv[2])
    adversary.to(DEVICE)

    print(f'Evaluating {sys.argv[1]} on k400val from the weights {sys.argv[2]}.\nConfiguartion: \nBATCH_SIZE={BATCH_SIZE}\nFRAMES{FRAME}')
    evaluate(adversary)
