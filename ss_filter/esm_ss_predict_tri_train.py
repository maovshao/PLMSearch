from __future__ import print_function,division

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from ss_filter_util.esm_ss_predict import esm_ss_dataset, esm_ss_predict_tri


def eval_ss(model, valid_dataloader, device):
    num_batches = len(valid_dataloader)
    model.eval()
    mse_loss_total = 0.0
    mae_loss_total = 0.0
    with torch.no_grad():
        for x0, x1, y in valid_dataloader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            y = y.to(device)
            y_pred = model(x0, x1)
            mse_loss_total += F.mse_loss(y_pred, y)
            mae_loss_total += F.l1_loss(y_pred, y)

    print(f"Test_mse_loss_avg: {mse_loss_total/num_batches:>7f}")
    print(f"Test_mae_loss_avg: {mae_loss_total/num_batches:>7f}")

def train_ss(model, train_dataloader, device, optimizer):
    size = len(train_dataloader.dataset)
    model.train()
    for batch, (x0, x1, y) in enumerate(train_dataloader):
        x0 = x0.to(device)
        x1 = x1.to(device)
        y = y.to(device)
        y_pred = model(x0, x1)
        loss_total = F.mse_loss(y_pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss_total, current = loss_total.item(), batch * len(x0)
            print(f"Train_mse_loss_avg: {loss_total:>7f}  [{current:>5d}/{size:>5d}]")

def main():
    parser = argparse.ArgumentParser('Script for training structure similarity prediction model')

    #input
    parser.add_argument('-mer', '--mean_esm_result', type=str, default='./ss_filter_data/esm_ss_predict/train/mean_esm_result.pkl', help="Mean esm result.")
    parser.add_argument('-il', '--input_list', type=str, default='./ss_filter_data/esm_ss_predict/train/protein_list.txt', help="Iutput protein list, decide the protein order of the ss_mat.")
    parser.add_argument('-im', '--input_mat', type=str, default='./ss_filter_data/esm_ss_predict/train/ss_mat.npz', help="Input mat name.")
    parser.add_argument('-d', '--device-id', default=[0], nargs='*', help='gpu device list, if only cpu then set it None or empty')

    #output
    parser.add_argument('--save_model_path', type=str, help='Pretrained MTPLM model for resuming training (optional)')

    # training parameters
    parser.add_argument('--ss_batch_size', type=int, default=100, help='minibatch size for ss loss (default: 100)')
    parser.add_argument('--epochs', type=int, default=20, help='number ot epochs (default: 20)')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate (default: 1e-5)')

    args = parser.parse_args()
    
    model = esm_ss_predict_tri(embed_dim = 1280)

    ## set the device
    if (args.device_id == None or args.device_id == []):
        print("None of GPU is selected.")
        device = "cpu"
        model.to(device)
        model_methods = model
    else:
        if torch.cuda.is_available()==False:
            print("GPU selected but none of them is available.")
            device = "cpu"
            model.to(device)
            model_methods = model
        else:
            print("We have", torch.cuda.device_count(), "GPUs in total! We will use as you selected")
            model = nn.DataParallel(model, device_ids = args.device_id)
            device = f'cuda:{args.device_id[0]}'
            model.to(device)
            model_methods = model.module

    print(f'# training with esm_ss_predict_tri: ss_batch_size={args.ss_batch_size}, epochs={args.epochs}, lr={args.lr}')
    print(f'# save model path: {args.save_model_path}')

    ## load the datasets
    # 1. ss
    ss_dataset = esm_ss_dataset(args.mean_esm_result, args.input_list, args.input_mat)
    ss_train_size = int(0.9 * len(ss_dataset))
    ss_valid_size = len(ss_dataset) - ss_train_size
    ss_train_dataset, ss_valid_dataset = torch.utils.data.random_split(ss_dataset, [ss_train_size, ss_valid_size])
  
    # iterators for the ss data
    batch_size = args.ss_batch_size
    ss_train_dataloader = torch.utils.data.DataLoader(ss_train_dataset
                                                    , batch_size=batch_size
                                                    , shuffle=True
                                                    )
    ss_valid_dataloader = torch.utils.data.DataLoader(ss_valid_dataset
                                                    , batch_size=batch_size
                                                    , shuffle=True
                                                    )

    params = [p for p in model_methods.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr = args.lr)

    ## train the model
    print('# training model', file=sys.stderr)

    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_ss(model, ss_train_dataloader, device, optimizer)
        eval_ss(model, ss_valid_dataloader, device)
    
    # save the model
    if args.save_model_path is not None:
        torch.save(model.module.state_dict(), args.save_model_path)

if __name__ == '__main__':
    main()
