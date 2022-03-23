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
from tqdm import tqdm

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


# FGSM Attack
def fgsm_attack(video, data_grad, epsilon = 0.0002):
    sign_data_grad = data_grad.sign()
    perturbed_video = video + epsilon * sign_data_grad
    return perturbed_video

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
train_ucf = datasets.Kinetics("../UCF101", split='train', frames_per_clip= FRAME, step_between_clips = 32, transform = train_transform, download=False, num_workers= 80)
train_hmdb51 = datasets.Kinetics("../hmdb51", split='train', frames_per_clip= FRAME, step_between_clips = 32, transform = train_transform, download=False, num_workers= 80)
train_ds = torch.utils.data.ConcatDataset([train_kinetics, train_ucf, train_hmdb51])
#train_ds = torch.utils.data.ConcatDataset([train_kinetics])
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
        if ct >= 7:
           ls2+=list(child.parameters())
        else:
           ls1+=list(child.parameters())
    optim1 = torch.optim.AdamW(ls1, lr=0.000003)
    optim2 = torch.optim.AdamW(ls2, lr=0.00003)
    sc1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optim1, factor=0.99, patience=45)
    sc2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optim2, factor=0.99, patience=45)
    criterion = nn.KLDivLoss(reduction = "mean")
    
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    #criterion = nn.MSELoss()

    for idx in range(10):
      print('\nStarting Epoch: {}\n'.format(idx))
      rloss = 0.0;
      model.train()
      for step,(video, label) in enumerate(train_dl):
          if step>-1:
             break
          torch.cuda.empty_cache()

          optim1.zero_grad()
          optim2.zero_grad()

          video = Variable(video.to(DEVICE), requires_grad=False)
          video = video.permute(0, 2, 1, 3, 4)
          video = size_changer(video, FRAME, 112)
          per_vid = video + torch.randn_like(video)

          label_ = victim(video)
          label_ = torch.nn.functional.gumbel_softmax(label_, tau=1, hard=False, eps=1e-10, dim=- 1)
          label_pervid = victim(per_vid)
          label_pervid = torch.nn.functional.gumbel_softmax(label_pervid, tau=1, hard=False, eps=1e-10, dim=- 1)
          video.requires_grad = True
          
          pred_pervid = model(per_vid)
          pred_pervid = torch.nn.functional.log_softmax(pred_pervid)
          pred = model(video)
          pred = torch.nn.functional.log_softmax(pred)
          ## Debugging 
          # print("##################################################################")
          # print("####   Prediction    #####")
          # print(pred, pred.size())
          # print("##################################################################")
          # print("####   Label        ######")  
          # print(label_, label_.size())
          # print("##################################################################")
          

          loss = criterion(pred,label_) + criterion(pred_pervid, label_pervid)
          rloss+=loss.item()
          loss.backward()
          vid_grad = video.grad.data
          adv_data = fgsm_attack(video, vid_grad)
          label_adv = victim(adv_data)
          pred_adv = model(adv_data) 
          label_adv = torch.nn.functional.gumbel_softmax(label_adv, tau=1, hard=False, eps=1e-10, dim=- 1)
          pred_adv = torch.nn.functional.log_softmax(pred_adv)

          adv_loss = criterion(pred_adv, label_adv)
          adv_loss.backward()
          rloss+=adv_loss.item()
          
          optim1.step()
          optim2.step()
          sc1.step(rloss/(step+1))
          sc2.step(rloss/(step+1))
          # print(f'Predicted class: {torch.argmax(pred, dim=1)}, Teacher class: {label_}, Actual label: {label}')
          print(rloss/(step+1), step)

          if (step % 100):
            torch.save(model, 'swin_to_r21_weights_prada.pth')

      print(f'avg loss: {rloss/len(train_dl)}')
      print('evaluation:')
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
              l_ = victim(video)
              l_ = torch.nn.functional.gumbel_softmax(l_, tau=1, hard=False, eps=1e-10, dim=- 1)
              prediction = model(video)
          # l_ = victim(video)
              print(f'Predicted class: {torch.argmax(prediction, dim=1)}, Teacher class: {torch.argmax(l_, dim=1)}, Actual label: {label}')
          # print(torch.argmax(prediction, dim=1), label)
          # print(f'Accuracy : {(torch.sum(torch.argmax(prediction, dim=1) == label)/len(label))*100.0}%')
          # print(f'Accuracy : {get_accuracy(torch.argmax(prediction, dim=1).tolist(), label)}')
              acc_list = accuracy(prediction.cpu(), torch.argmax(l_, dim=1).cpu(), topk=(1,5))
              acc1.append(acc_list[0])
              acc5.append(acc_list[1])
              print('top1 =',torch.mean(torch.stack(acc1)))
              print('top5 =',torch.mean(torch.stack(acc5)))

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
    
    #r2 = torch.hub.load("moabitcoin/ig65m-pytorch", "r2plus1d_34_32_ig65m", num_classes=359, pretrained=True)
    #r2.fc = nn.Linear(in_features=512, out_features=400, bias=True)
    adversary = torch.load("r21_weights.pth")

    adversary.to(DEVICE)
    
    train_with_extraction(adversary, victim)
