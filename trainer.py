import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import AverageMeter, char_color


class Trainer(object):

    def __init__(self, model, train_loader, val_loader, args, device, logging):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = device
        self.logging = logging

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5,
                                                                    patience=3, verbose=True, min_lr=1e-5)
        if args.action == 'train':
            self.writer = SummaryWriter(log_dir=args.tensorboard_dir)
            self.inputs = next(iter(train_loader))[0]
            self.writer.add_graph(model, self.inputs.to(device, dtype=torch.float32))
        if args.DataParallel:
            self.model = torch.nn.DataParallel(model)
        else:
            self.model = model

    def train(self):
        epochs = self.args.epochs
        n_train = len(self.train_loader.dataset)
        step = 0
        best_acc = 0.
        accs = AverageMeter()
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            # training
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                for batch in self.train_loader:
                    images, labels = batch[0], batch[1]

                    images = images.to(device=self.device, dtype=torch.float32)
                    labels = labels.to(device=self.device, dtype=torch.long)
                    preds = self.model(images)
                    loss = self.criterion(preds, labels)

                    epoch_loss += loss.item()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    accs.update((preds.argmax(1) == labels).sum().item() / images.size(0), images.size(0))
                    pbar.set_postfix(**{'loss': loss.item(), 'acc': accs.avg})
                    self.writer.add_scalar('acc/train', accs.avg, step)
                    self.writer.add_scalar('Loss/train', loss.item(), step)
                    pbar.update(images.shape[0])
                    step = step + 1
            # eval
            if (epoch + 1) % self.args.val_epoch == 0:
                acc = self.test(mode='val')
                if acc > best_acc:
                    best_acc = acc
                    if self.args.save_path:
                        if not os.path.exists(self.args.save_path):
                            os.makedirs(self.args.save_path)
                        torch.save(self.model.state_dict(), f'{self.args.save_path}/best_model.pth')
                        self.logging.info(char_color(f'best model saved !', word=33))

                self.logging.info(f'acc: {acc}')
                self.writer.add_scalars('Valid', {'acc': acc}, step)
                self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], step)
                self.scheduler.step(acc)
            if (epoch + 1) % self.args.save_model_epoch == 0:
                if self.args.save_path:
                    if not os.path.exists(self.args.save_path):
                        os.makedirs(self.args.save_path)
                    model_name = f'{self.args.task}_'
                    torch.save(self.model.state_dict(), f'{self.args.save_path}/{model_name}{epoch + 1}.pth')
                    self.logging.info(char_color(f'Checkpoint {epoch + 1} saved !'))
        self.writer.close()

    def test(self, mode='val', model_path=None, aug=False):
        self.model.train(False)
        self.model.eval()

        accs = AverageMeter()
        test_len = len(self.val_loader)
        step = 0
        with torch.no_grad():
            with tqdm(total=test_len, desc=f'{mode}', unit='batch') as pbar:
                for batch in self.val_loader:
                    images, labels = batch[0], batch[1]
                    images = images.to(device=self.device, dtype=torch.float32)
                    labels = labels.to(device=self.device, dtype=torch.long)
                    preds = self.model(images)
                    accs.update((preds.argmax(1) == labels).sum().item() / images.size(0), images.size(0))
                    pbar.set_postfix(**{'acc': accs.avg})
                    pbar.update(images.shape[0])
                    step = step + 1
        return accs.avg
