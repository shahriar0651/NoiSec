import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from helper import *
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import sys

def train(model, data_loader, criterion, optimizer, loss_mode):
    running_loss = 0
    model.train()
    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
        batch_y = batch_y.squeeze().long()
        print(f"Batch: {step}, {batch_x.shape}, {batch_y.shape}")
        optimizer.zero_grad()
        output = model(batch_x) # get predict label of batch_x

        if loss_mode == "mse":
            loss = criterion(output, batch_y) # mse loss
        elif loss_mode == "cross":
            loss = criterion(output, torch.argmax(batch_y, dim=1)) # cross entropy loss

        loss.backward()
        optimizer.step()
        running_loss += loss
    return running_loss


def eval_model(cfg, model, data_loader, device):
    model.eval() 
    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_y = batch_y.squeeze().long()
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_y_predict = model(batch_x)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_predict.append(batch_y_predict)
        batch_y = torch.argmax(batch_y, dim=1)
        y_true.append(batch_y)
    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)


    return accuracy_score(y_true.cpu(), y_predict.cpu())


def train_one_epoch(cfg, net, optimizer, scheduler, loss_fnc, criterion, train_loader, device):
    net.train()
    running_loss = 0
    batch_loss = 0
    print_frequency = 1000
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # print("labels : ", labels)
        # labels = labels.squeeze().long()
        optimizer.zero_grad()

        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        
        if loss_fnc == 'mse':
            loss = criterion(outputs, labels)
        elif loss_fnc == 'cross':
            loss = criterion(outputs, torch.argmax(labels, dim=1))
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()  # .item() to get the scalar value of the loss
        batch_loss += loss.item()  # .item() to get the scalar value of the loss
        
        
        # ------------ Commenting for now --------------------
        # Update Learning Rate
        # Record & update learning rate
        # for param_group in optimizer.param_groups:
        #     print(param_group['lr'])
        #     break
        # scheduler.step()
        # --------------------------------------------------

        if batch_idx % print_frequency == 0 and batch_idx > 0:
            sys.stdout.write(f'\rBatch [{batch_idx}/{len(train_loader)}], Loss: {batch_loss/print_frequency:.4f}')
            sys.stdout.flush()
            batch_loss = 0
    return net, running_loss

def train_conv_net(cfg, net, model_type, train_loader, test_loader_cln, test_loader_trg):
    device = torch.device(cfg.device)
    net.to(device)
    
    if model_type == 'badnet':
        print("Starting BadNet Model")
        if cfg.dataset.name == 'mnist':
            loss_fnc = 'mse' #FIXME : Overwriting for BadNet
            lr = 0.01
            EPOCHS = 15
        elif cfg.dataset.name in ['cifar10', 'gtsrb']:
            loss_fnc = 'mse' #FIXME : Overwriting for BadNet
            lr = 0.01
            EPOCHS = cfg.dataset.epochs
        else:
            lr = 0.001
            EPOCHS = cfg.dataset.epochs
        
        loss_fnc = 'mse' #FIXME : Overwriting for BadNet
        optim_alg = 'sgd'
    else:
        loss_fnc = cfg.dataset.loss_fn
        optim_alg = cfg.dataset.optim
        lr = cfg.dataset.lr
        EPOCHS = cfg.dataset.epochs
        
    print(f'Starting with {loss_fnc, optim_alg, lr} for {model_type}')
    
    loss_fn_dict = {
        'mse' : nn.MSELoss(),
        'cross' : nn.CrossEntropyLoss()
        }
    optim_dict = {
        'adam' : optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.95)),
        'sgd' : optim.SGD(net.parameters(), lr=lr, momentum=0.9) #, weight_decay=5e-4)
        }
    
    criterion = loss_fn_dict[loss_fnc]
    optimizer =  optim_dict[optim_alg]
    poison_ratio = cfg.dataset.poison_ratio if model_type=='badnet' else 0
    trigger_label = cfg.dataset.trigger_label if model_type=='badnet' else 0
    num_feats = cfg.dataset.num_feats

    #Added 
    if cfg.scheduler:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr*100, epochs=EPOCHS, 
                                                steps_per_epoch=len(train_loader))
    else:
        scheduler = None
    
    best_loss = float('inf')
    counter = 0
    # Early stopping parameters
    patience = 50
    epochs_without_improvement = 0
    best_val_acc = 0


    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        net, running_loss = train_one_epoch(cfg, net, optimizer, scheduler, loss_fnc, criterion, train_loader, device)
        if cfg.scheduler:
            scheduler.step()
        acc_train_trg = eval_model(cfg, net, train_loader, device)
        acc_test_cln = eval_model(cfg, net, test_loader_cln, device)
        acc_test_trg = eval_model(cfg, net, test_loader_trg, device) if model_type=='badnet' else -100

        print("# EPOCH-%d   loss: %.4f  training acc: %.4f, clean testing acc: %.4f trigger testing acc: %.4f\n"\
              % (epoch, running_loss, acc_train_trg, acc_test_cln, acc_test_trg))
        
        
        # Early stopping check
        if acc_test_cln > best_val_acc:
            best_val_acc = acc_test_cln
            epochs_without_improvement = 0
            # Save the best model
            model_dir = Path(f'{cfg.models_dir}/{cfg.dataset.name}_net_{cfg.cls_type}_{model_type}_{num_feats}_{trigger_label}_{poison_ratio}_best.pth')
            torch.save(net.state_dict(), model_dir)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered!")
                break
        
        # # Check for early stopping
        # if running_loss < best_loss:
        #     best_loss = running_loss
        #     counter = 0
        # else:
        #     counter += 1
        #     if counter >= patience:
        #         print(f"Early stopping after {epoch+1} epochs without improvement.")
        #         break
        # if ((acc_train_trg > 0.98) and (acc_test_cln > 0.98)) or (acc_test_trg < 0.0 or acc_test_trg > 0.98):
        #     print(f"Accuracy achieved!!! Breaking")
        #     break
    print('Finished Training')
    
    model_dir = Path(f'{cfg.models_dir}/{cfg.dataset.name}_net_{cfg.cls_type}_{model_type}_{num_feats}_{trigger_label}_{poison_ratio}_{cfg.dataset.epochs}.pth')
    torch.save(net.state_dict(), model_dir)
    print('Movel Saved')
    return net
