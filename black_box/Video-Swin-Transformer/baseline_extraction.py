import warnings
warnings.filterwarnings('ignore')
from ignite.metrics import Accuracy, Loss
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import torch.nn.functional as func
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch
from mmaction.datasets import build_dataloader, build_dataset
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmaction.models import build_model
from mmcv import Config, DictAction

config = '/content/Video-Swin-Transformer/configs/recognition/swin/swin_base_patch244_window877_kinetics400_1k.py'
checkpoint = '/content/swin_base_patch244_window877_kinetics400_1k.pth'


DEVICE = torch.device('cuda:0')
SPATIAL_DIM = 224
TEMPORAL_DIM = 16
NUM_CHANNELS = 3


class QueriedDatasetFromVictim(Dataset):
    def __init__(self, model, s):
        super().__init__()
        self.size = s
        self.X = torch.randn((s, NUM_CHANNELS, TEMPORAL_DIM, SPATIAL_DIM, SPATIAL_DIM)).to(DEVICE)
        self.y_queried = []
        # querying predictions one by one since colab gpu is not sufficient for parallel load
        for x in self.X:
            outputs = model(torch.tensor(x)[None, :]).to(DEVICE)
            self.y_queried.append(torch.argmax(outputs, dim=1))
        self.y_queried = torch.tensor(self.y_queried)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.X[index], self.y_queried[index]


def train_with_extraction(model, victim):
    # again, batch_size=1 due to compute restrictions on colab
    train_batch_size, test_batch_size = 1, 1

    print('\nloading data')
    train_dl = DataLoader(QueriedDatasetFromVictim(victim, 100), train_batch_size, shuffle=True)
    # to change: test_dl --> kinetics400 dataloader
    test_dl = DataLoader(QueriedDatasetFromVictim(victim, 10), test_batch_size, shuffle=False)
    print('data loaded\n')

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    criterion = nn.CrossEntropyLoss()

    trainer = create_supervised_trainer(model, optimizer, criterion, device=DEVICE)
    val_metrics = {"accuracy": Accuracy(), "nll": Loss(criterion)}
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=DEVICE)

    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def log_test_results(trainer):
        print(f"\nTest Results - Epoch: {trainer.state.epoch}")
        evaluator.run(test_dl)
        metrics = evaluator.state.metrics
        print(f"Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

    @evaluator.on(Events.ITERATION_COMPLETED(every=1))
    def print_output(evaluator):
        pred, actual = evaluator.state.output
        print(f'Predicted class: {torch.argmax(pred, dim=1).item()}, Actual label: {actual.item()}')

    print('starting training')
    trainer.run(train_dl, max_epochs=100)


# wrapper around mmactions Recognizer3d class to provide nn.Module like interface
# (for compatibility with ignite methods)
class MMActionModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X):
        return self.model.forward_dummy(X)[0]


if __name__ == '__main__':
    cfg = Config.fromfile(config)
  
    model_victim = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    # loading pretrained weights to victim
    load_checkpoint(model_victim, checkpoint, map_location='cuda:0')
    model_victim.to(DEVICE)
    victim = MMActionModelWrapper(model_victim)

    model_adversary = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model_adversary.to(DEVICE)
    adversary = MMActionModelWrapper(model_adversary)

    train_with_extraction(adversary, victim)
