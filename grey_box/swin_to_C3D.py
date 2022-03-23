# Imports for transform and dataset prepration
import numpy as np
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.utils.data import TensorDataset, DataLoader, Dataset, WeightedRandomSampler
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import torch.nn.functional as func
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch
from torch.autograd import Variable

from c3d_pytorch.C3D_model import C3D

# Train transform and other utils
batch_size = 16
FRAME = 32

def tofloat(x):
  return x[:FRAME].float()
  # return x.float()


train_transform = transforms.Compose([
                    # transforms.Lambda(lambda x: x / 255.0),
                    #transforms.functional.uniform_temporal_subsample_repeated(32, (1,0), temporal_dim = 2),
                    tofloat,
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.Normalize((123, 116, 103), (58, 57, 57)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=25/225),
                    transforms.RandomRotation(15)
                  ])


test_transform = transforms.Compose([
                    # transforms.Lambda(lambda x: x / 255.0),
                    #transforms.functional.uniform_temporal_subsample_repeated(32, (1,0), temporal_dim = 2),
                    tofloat,
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.Normalize((123, 116, 103), (58, 57, 57)),
                    #transforms.RandomHorizontalFlip(p=0.5),
                    #transforms.ColorJitter(brightness=25/225),
                    #transforms.RandomRotation(15)
                  ])


# Load Dataset

def collate_fn(batch):
    # print(batch[:10])
    x = torch.stack([torch.tensor(data_item[0]) for data_item in batch])
    y = [int(data_item[2]) for data_item in batch]
    # return x[:32], y
    return x, y
    
DUMMY = '../k400val_dummy'
VALIDATION = '../k400val_pytorch'
TRAIN = '../kinetics400_5per'

train_kinetics = datasets.Kinetics(TRAIN, frames_per_clip= FRAME, split='train', num_classes= '600', step_between_clips= FRAME*2, transform = train_transform,  download= False, num_download_workers= 1, num_workers= 80)
test_kinetics = datasets.Kinetics(VALIDATION, frames_per_clip= FRAME, split='val', num_classes= '600', step_between_clips= 2000000, transform = test_transform,  download= False, num_download_workers= 1, num_workers= 80)
train_ds = train_kinetics
test_ds = test_kinetics
train_dl = DataLoader(train_ds, collate_fn=collate_fn, batch_size = batch_size, shuffle = True);
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

config = 'configs/recognition/swin/swin_base_patch244_window877_kinetics400_1k.py'
checkpoint = '../swin_base_patch244_window877_kinetics400_1k.pth'

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


def train_with_extraction(model, victim):
    # again, batch_size=1 due to compute restrictions on colab
    ct = 0
    ls1 = []
    ls2 = []
    for child in model.children():
        ct += 1
        if ct >= 14:
           ls2+=list(child.parameters())
        else:
           ls1+=list(child.parameters())
    optim1 = torch.optim.AdamW(ls1, lr=0.00003)
    optim2 = torch.optim.AdamW(ls2, lr=0.0003)
    criterion = nn.KLDivLoss()

    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    #criterion = nn.MSELoss()

    for idx in range(10):
      print('\nStarting Epoch: {}\n'.format(idx))
      rloss = 0.0;
      model.train()
      for step,(video, label) in enumerate(train_dl):
          #if step>1:
          #    break
          torch.cuda.empty_cache()

          optim1.zero_grad()
          optim2.zero_grad()
          
          video = Variable(video.to(DEVICE), requires_grad=False)
          video = video.permute(0, 2, 1, 3, 4)
          label_ = victim(video)
          label_ = torch.nn.functional.gumbel_softmax(label_, tau=1, hard=False, eps=1e-10, dim=- 1)
          
          video = size_changer(video)
          pred = model(video)
          pred = torch.log(pred)

          ## Debugging 
          # print("##################################################################")
          # print("####   Prediction    #####")
          # print(pred, pred.size())
          # print("##################################################################")
          # print("####   Label        ######")  
          # print(label_, label_.size())
          # print("##################################################################")
          

          loss = criterion(pred,label_)
          rloss+=loss.item()
          loss.backward()
          optim1.step()
          optim2.step()
          # print(f'Predicted class: {torch.argmax(pred, dim=1)}, Teacher class: {label_}, Actual label: {label}')
          print(rloss/(step+1), step)
          if (step % 100):
              torch.save(model, 'swin_to_c3d_weights.pth')

      print(f'avg loss: {rloss/len(train_dl)}')
      print('evaluation:')
      model.eval()
      with torch.no_grad():
          acc = []
          for step,(video, label) in enumerate(test_dl):
              #if step > 5:
              #    break
              video = Variable(video.to(DEVICE), requires_grad=False)
              video = video.permute(0, 2, 1, 3, 4)
              l_ = victim(video)
              l_ = torch.nn.functional.gumbel_softmax(l_, tau=1, hard=False, eps=1e-10, dim=- 1)
              video = size_changer(video)
              prediction = model(video)
              prediction = torch.log(prediction)
          # l_ = victim(video)
              # print(f'Predicted class: {torch.argmax(prediction, dim=1)}, Teacher class: {torch.argmax(l_, dim=1)}, Actual label: {label}')
          # print(torch.argmax(prediction, dim=1), label)
          # print(f'Accuracy : {(torch.sum(torch.argmax(prediction, dim=1) == label)/len(label))*100.0}%')
          # print(f'Accuracy : {get_accuracy(torch.argmax(prediction, dim=1).tolist(), label)}')
              acc.append(topk(prediction.cpu(), torch.argmax(l_, dim=1).cpu(), 1))
      print(np.mean(acc))


cfg = Config.fromfile(config)
if __name__ == '__main__':
    model_victim = build_model(cfg.model, train_cfg=None, test_cfg=None)
    # loading pretrained weights to victim
    load_checkpoint(model_victim, checkpoint, map_location=DEVICE)
    model_victim.to(DEVICE)
    victim = MMActionModelWrapper(model_victim)
    for param in victim.parameters():
      param.requires_grad = False
    victim.eval()

    # net = C3D()
    # net.load_state_dict(torch.load('c3d.pickle'))
    # net.fc8 = nn.Linear(in_features=4096, out_features=400, bias=True)
    size_changer = torch.nn.AvgPool3d((1, 2, 2), stride=None, padding=0, ceil_mode=False)
    # adversary = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
    # print(net)
    # model_adversary = build_model(cfg.model, train_cfg=None, test_cfg=None)
    saved_weights = 'c3d_weights.pth'
    net = torch.load(saved_weights)
    adversary = net
    adversary.to(DEVICE)
    train_with_extraction(adversary, victim)
