import torch
import os
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from statistics import mean
from tqdm import tqdm

from .model import ARCNNEmbedding
from .data_utils import ESDataset, Collate
from .loss import ContrastiveLoss
from .utils import (
    Hparam,
    save_checkpoint,
    load_checkpoint,
)

def main():
    hparams = Hparam("./config.yaml")
    train(hparams=hparams)

def train(hparams):
    global_step = 0
    global_eval_loss = 0

    if not hparams.train.use_gpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ESDataset(hparams, hparams.train.train_speaker_id)
    train_loader = DataLoader(train_dataset, batch_size=hparams.train.train_batch_size, 
        num_workers=4, shuffle=True, collate_fn=Collate())

    eval_dataset = ESDataset(hparams, hparams.train.eval_speaker_id)
    eval_loader = DataLoader(eval_dataset, batch_size=hparams.train.eval_batch_size,
        num_workers=4, shuffle=True, collate_fn=Collate())
    
    model = ARCNNEmbedding(hparams=hparams).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=hparams.train.learning_rate,
                            betas=(0.9, 0.99), weight_decay=5e-4)
    
    start_epoch = 0
    if os.path.isfile(hparams.checkpoint.continue_once):
        model, optimizer, start_epoch = load_checkpoint(hparams.checkpoint.continue_once, model, optimizer)

    criterion = ContrastiveLoss()
    
    model.train()
    for epoch in range(start_epoch, hparams.train.num_epochs):
        losses = []
        with tqdm(train_loader, unit="batch") as tepoch:
            for x0, x1, y in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                x0 = x0.to(device=device, dtype=torch.float32)
                x1 = x1.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)
                optimizer.zero_grad()
                anchor_outputs = model(x0)
                another_outputs = model(x1)
                loss = criterion(anchor_outputs, another_outputs, y)
                losses.append(loss.item())

                loss.backward()
                optimizer.step()
                
                if global_step % hparams.checkpoint.valid_interval == 0:
                    valid_loss = evaluate(model, eval_loader, criterion, device=device)
                    if valid_loss >= global_eval_loss:
                        global_eval_loss = valid_loss
                        
                        if global_step != 0:
                            if not os.path.isdir(hparams.checkpoint.save_folder):
                                os.mkdir(hparams.checkpoint.save_folder)

                            checkpoint_path = os.path.join(hparams.checkpoint.save_folder, 
                                                            "model_{}_{}.pt".format(epoch, round(global_eval_acc, 2)))
                            save_checkpoint(model, optimizer, epoch, checkpoint_path)

                global_step += 1
                tepoch.set_postfix(train_loss=mean(losses), valid_loss=global_eval_loss)             


def evaluate(model, eval_loader, criterion, device='cpu'):
    model.eval()
    with torch.no_grad():
        losses = []
        for x0, x1, y in eval_loader:
            
            x0 = x0.to(device=device, dtype=torch.float32)
            x1 = x1.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            
            anchor_outputs = model(x0)
            another_outputs = model(x1)
            loss = criterion(anchor_outputs, another_outputs, y)
            losses.append(loss.item())

        loss = mean(losses)
        model.train()

        return loss


if __name__ == "__main__":
    main()