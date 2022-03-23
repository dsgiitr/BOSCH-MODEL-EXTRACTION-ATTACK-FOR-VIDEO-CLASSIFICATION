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

#from movinets import MoViNet
#from movinets.config import _C
#from c3d_pytorch.C3D_model import C3D

#from pytorchvideo.transforms import (
#    ApplyTransformToKey,
#    Normalize,
#    RandomShortSideScale,
#    RemoveKey,
#    ShortSideScale,
#    UniformTemporalSubsample
#)

def topk(output, target, maxk=5):
    """Computes the precision@k for the specified value of maxk"""
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = correct[:maxk].view(-1).float().sum(0)
    return correct_k.mul_(100.0 / batch_size)


# Train transform and other utils

from torchvision.transforms.transforms import ToTensor
data_flip = transforms.Compose([
    transforms.Resize((28, 28)),
     transforms.ToTensor(),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.Normalize(0,1)
     #AddGaussianNoise(0., 10/255)
])

# Resize, normalize and rotate image
data_rotate = transforms.Compose([
	transforms.Resize((28, 28)),
   transforms.ToTensor(),
    transforms.RandomRotation(30),
    transforms.Normalize(0,1)
])

test_transform = transforms.Compose([
     transforms.Resize((28, 28)),
     transforms.ToTensor(),
     transforms.Normalize(0,1)
])


# Resize, normalize and shear image
data_shear = transforms.Compose([
	transforms.Resize((28, 28)),
   transforms.ToTensor(),
    transforms.RandomAffine(degrees = 15,shear=2),
    transforms.Normalize(0,1)
])

# Resize, normalize and translate image
data_translate = transforms.Compose([
	transforms.Resize((28, 28)),
   transforms.ToTensor(),
    transforms.RandomAffine(degrees = 15,translate=(0.1,0.1)),
    transforms.Normalize(0,1)
])

# Resize, normalize and crop image
data_center = transforms.Compose([
	  transforms.Resize((36, 36)),
    transforms.CenterCrop(28),
     transforms.ToTensor(),
    transforms.Normalize(0,1)
])
# Resize, normalize and jitter image brightness
data_jitter_brightness = transforms.Compose([
	  transforms.Resize((28, 28)),
     transforms.ToTensor(),
    transforms.ColorJitter(brightness=5),
    transforms.Normalize(0,1)
])

# Resize, normalize and jitter image saturation
data_jitter_saturation = transforms.Compose([
	  transforms.Resize((28, 28)),
     transforms.ToTensor(),
    transforms.ColorJitter(saturation=5),
    transforms.Normalize(0,1)
])

# Resize, normalize and jitter image contrast
data_jitter_contrast = transforms.Compose([
	  transforms.Resize((28, 28)),
     transforms.ToTensor(),
    transforms.ColorJitter(contrast=5),
    transforms.Normalize(0,1)
    ])

# Resize, normalize and jitter image hues
data_jitter_hue = transforms.Compose([
	transforms.Resize((28, 28)),
   transforms.ToTensor(),
    transforms.ColorJitter(hue=0.4),
    transforms.Normalize(0,1)
])

batch_size = 16
FRAME = 16

def tofloat(x):
  return x[:FRAME].float()
class shift():
  def __init__(self, sz):
    self.sz = sz
  def __call__(self, x):
    return torch.permute(x, self.sz)
def pr(x):
  print(x.shape)
  return x

train_transform = transforms.Compose([
                    #transforms.Lambda(lambda x: x / 255.0),
                    #transforms.functional.uniform_temporal_subsample_repeated(32, (1,0), temporal_dim = 2),
                    transforms.ToTensor(),
                    transforms.CenterCrop(224),
                    transforms.Resize(224),
                    transforms.Normalize((123, 116, 103), (58, 57, 57)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=25/225),
                    transforms.RandomRotation(15)
                  ])


test_transform = transforms.Compose([
                    #transforms.Lambda(lambda x: x / 255.0),
                    #transforms.functional.uniform_temporal_subsample_repeated(32, (1,0), temporal_dim = 2),
                    tofloat,
                    transforms.CenterCrop(224),
                    transforms.Resize(224),
                    transforms.Normalize((123, 116, 103), (58, 57, 57)),
                    # transforms.RandomHorizontalFlip(p=0.5),
                    # transforms.ColorJitter(brightness=25/225),
                    # transforms.RandomRotation(15)
                  ])


from torchvision.transforms import RandomAffine
random_affine = RandomAffine(15, (0.1, 0.1), shear=10)

def image_to_vid(image, frames=16):
    video = image.unsqueeze(2).expand(-1, -1, 32, -1, -1)
    for i in range(frames):
        video[:,:,i] = random_affine(video[:,:,i])
    return video



# Load Dataset

def collate_fn(batch):
    # print(batch[:10])
    x = torch.stack([torch.tensor(data_item[0]) for data_item in batch])
    y = [int(data_item[2]) for data_item in batch]
    # return x[:32], y
    return x, y
    
DUMMY = '../../../k400val_dummy'
VALIDATION = '../../../k400val_pytorch'
TRAIN = '../../../coco_person_dataset'


train_ds = datasets.ImageFolder(TRAIN, train_transform)
# train_kinetics = datasets.Kinetics(TRAIN, frames_per_clip= FRAME, split='train', num_classes= '400', step_between_clips= FRAME*2, transform = train_transform,  download= False, num_download_workers= 1, num_workers= 80)
test_kinetics = datasets.Kinetics(VALIDATION, frames_per_clip= FRAME, split='val', num_classes= '400', step_between_clips= 2000000, transform = test_transform,  download= False, num_download_workers= 1, num_workers= 80)
#train_ucf = datasets.Kinetics("../UCF101", split='train', frames_per_clip= FRAME, step_between_clips = 16, transform = train_transform, download=False, num_workers= 80)
#train_hmdb51 = datasets.Kinetics("../hmdb51", split='train', frames_per_clip= FRAME, step_between_clips = 16, transform = train_transform, download=False, num_workers= 80)
#train_ds = torch.utils.data.ConcatDataset([train_kinetics, train_ucf, train_hmdb51])
# train_ds = torch.utils.data.ConcatDataset([train_kinetics])
test_ds = test_kinetics
train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True);
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
checkpoint = '../../../swin_base_patch244_window877_kinetics400_1k.pth'

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

def get_accuracy(pred, actual):
    correct_labels = 0
    # print(f'length of pred {len(pred)}', pred, actual)
    for i in range(len(pred)):
        if (pred[i]==actual[i]):
            correct_labels+=1
    return (correct_labels/len(pred))*100.0

def size_changer(x, tm, sz):
      return torch.nn.functional.upsample(x, size=(tm,sz,sz), scale_factor=None, mode='nearest', align_corners=None)


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
    optim1 = torch.optim.AdamW(ls1, lr=0.00003)
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
      for step,(image, label) in enumerate(train_dl):
        #   if step>-1:
        #     break
          torch.cuda.empty_cache()

          optim1.zero_grad()
          optim2.zero_grad()
          
          image = Variable(image.to(DEVICE), requires_grad=False)
          video = image_to_vid(image, FRAME)
          label_ = victim(video)
          label_ = torch.nn.functional.gumbel_softmax(label_, tau=1, hard=False, eps=1e-10, dim=- 1)
          
          video = size_changer(video, FRAME, 112)
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
          

          loss = criterion(pred,label_)
          rloss+=loss.item()
          loss.backward()
          optim1.step()
          optim2.step()
          sc1.step(rloss/(step+1))
          sc2.step(rloss/(step+1))
          # print(f'Predicted class: {torch.argmax(pred, dim=1)}, Teacher class: {label_}, Actual label: {label}')
          print(rloss/(step+1), step)

          if (step % 100):
            torch.save(model, 'stacked_images_with_swin.pth')

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
              l_ = victim(video)
              l_ = torch.nn.functional.gumbel_softmax(l_, tau=1, hard=False, eps=1e-10, dim=- 1)
              video = size_changer(video, FRAME, 112)
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
    

    adversary = torch.load("../../../final_submission_grey/r21_weights.pth")
    adversary.to(DEVICE)
    
    train_with_extraction(adversary, victim)