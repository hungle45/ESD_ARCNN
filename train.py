import torch
import os
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from statistics import mean
from tqdm import tqdm

from .model import ARCNNEmbedding
from .data_utils import DataCollator
from .loss import ContrastiveLoss
from .utils import (
    Hparam,
    save_checkpoint,
    load_checkpoint,
)

def main():
    hparams = Hparam("./hparams.yaml")
    train(hparams=hparams)

def train(hparams, train_dataset, eval_dataset):
    global_step = 0
    global_eval_loss = float('inf')
    global_eval_accuracy = 0.0
    loss_coeff = hparams.train.loss_coeff

    if not hparams.train.use_gpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, batch_size=hparams.train.train_batch_size, 
        num_workers=hparams.data.num_workers, shuffle=True, 
        collate_fn=DataCollator(len(hparams.data.classes)))

    eval_loader = DataLoader(eval_dataset, batch_size=hparams.train.eval_batch_size,
        num_workers=hparams.data.num_workers, shuffle=True, 
        collate_fn=DataCollator(len(hparams.data.classes)))
    
    model = ARCNNEmbedding(hparams=hparams).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=hparams.train.learning_rate,
                            betas=(0.9, 0.99), weight_decay=5e-4)
    
    start_epoch = 0
    if os.path.isfile(hparams.checkpoint.continue_once):
        model, optimizer, start_epoch = load_checkpoint(hparams.checkpoint.continue_once, model, optimizer)

    classify_loss = torch.nn.CrossEntropyLoss()
    contrastive_loss = ContrastiveLoss(hparams.loss.margin, hparams.loss.mode)
    
    model.train()
    for epoch in range(start_epoch, hparams.train.num_epochs):
        losses = []
        with tqdm(train_loader, unit="batch") as tepoch:
            for x1, x2, y, l1, l2 in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                x1 = x1.permute(0,3,1,2).to(device=device, dtype=torch.float32)
                x2 = x2.permute(0,3,1,2).to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)
                l1 = l1.to(device=device, dtype=torch.float32)
                l2= l2.to(device=device, dtype=torch.float32)
                
                optimizer.zero_grad()
                outputs1 = model(x1)
                outputs2 = model(x2)
                loss1 = contrastive_loss(outputs1[0], outputs2[0], y) # embed
                loss2 = classify_loss(outputs1[1], l1) + classify_loss(outputs2[1], l2)
                loss = loss_coeff * loss1 + (1 - loss_coeff) * loss2
                losses.append(loss.item())

                loss.backward()
                optimizer.step()
                
                if global_step % hparams.checkpoint.valid_interval == 0:
                    valid_loss, valid_accuracy = evaluate(model, eval_loader, 
                        contrastive_loss, classify_loss, loss_coeff, device=device)
                    if valid_accuracy >= global_eval_accuracy:
                        global_eval_loss = valid_loss
                        global_eval_accuracy = valid_accuracy
                        
                        if global_step != 0:
                            if not os.path.isdir(hparams.checkpoint.save_folder):
                                os.mkdir(hparams.checkpoint.save_folder)

                            checkpoint_path = os.path.join(hparams.checkpoint.save_folder, 
                                                            "model_{}_{}.pt".format(epoch, round(global_eval_accuracy, 4)))
                            save_checkpoint(model, optimizer, epoch, checkpoint_path)

                global_step += 1
                tepoch.set_postfix(train_loss=mean(losses), valid_loss=valid_loss, valid_acc=valid_accuracy, 
                                   best_valid_loss=global_eval_loss, best_valid_acc=global_eval_accuracy)                     


def evaluate(model, eval_loader, contrastive_loss, classify_loss, loss_coeff=0.5, device='cpu'):
    model.eval()
    with torch.no_grad():
        losses = []
        y_ground_truth = []
        y_predict = []
        for x1, x2, y, l1, l2 in eval_loader:
            
            x1 = x1.permute(0,3,1,2).to(device=device, dtype=torch.float32)
            x2 = x2.permute(0,3,1,2).to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            l1 = l1.to(device=device, dtype=torch.float32)
            l2= l2.to(device=device, dtype=torch.float32)
            
            outputs1 = model(x1)
            outputs2 = model(x2)
            
            # loss
            loss1 = contrastive_loss(outputs1[0], outputs2[0], y) # embed
            loss2 = classify_loss(outputs1[1], l1) + classify_loss(outputs2[1], l2)
            loss = loss_coeff * loss1 + (1 - loss_coeff) * loss2
            losses.append(loss.item())

            # accuracy
            anchor_idxs_pred = torch.argmax(outputs1[1], dim=1)
            y_predict = y_predict + list(anchor_idxs_pred.cpu().detach().numpy())
                
            anchor_idxs_gt = torch.argmax(l1, dim=1)
            y_ground_truth = y_ground_truth + list(anchor_idxs_gt.cpu().detach().numpy())
            
        loss = mean(losses)
        correct = (np.array(y_ground_truth) == np.array(y_predict))
        accuracy = correct.sum() / correct.size
    model.train()
    
    return loss, accuracy


if __name__ == "__main__":
    main()