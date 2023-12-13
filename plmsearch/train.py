import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from logzero import logger
from scipy.stats import pearsonr, spearmanr

from plmsearch_util.model import plmsearch
from plmsearch_util.train_util import plmsearch_dataset
from plmsearch_util.util import make_parent_dir

def eval_ss(model, valid_dataloader, device, f):
    model.eval()
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for x0, x1, y in valid_dataloader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            y = y.to(device)

            y_pred = model.pairwise_predict(x0, x1)

            y_true_list.append(y)
            y_pred_list.append(y_pred)

        y_true = torch.cat(y_true_list)
        y_pred = torch.cat(y_pred_list)

        l1_loss = F.l1_loss(y_true, y_pred)
        l2_loss = F.mse_loss(y_true, y_pred)
        print(f"L1 loss: {l1_loss:.8f}")
        print(f"L2 loss: {l2_loss:.8f}")

        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pearson_r, _ = pearsonr(y_true, y_pred)
        spearman_r, _ = spearmanr(y_true, y_pred)
        
        print(f"Pearson correlation: {pearson_r:.8f}")
        print(f"Spearman correlation: {spearman_r:.8f}")

        f.write(f"{l1_loss:.8f}\t{l2_loss:.8f}\t{pearson_r:.8f}\t{spearman_r:.8f}\n")
    return spearman_r

def train_ss(model, train_dataloader, device, optimizer):
    model.train()
    for batch, (x0, x1, y) in enumerate(train_dataloader):
        x0 = x0.to(device)
        x1 = x1.to(device)
        y = y.to(device)

        y_pred = model.pairwise_predict(x0, x1)

        loss_total = F.mse_loss(y_pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

def main():
    parser = argparse.ArgumentParser('Script for training structure similarity prediction model')

    #input
    parser.add_argument('-ee', '--esm_embedding', type=list, default=['./plmsearch_data/train/scope40_train/embedding.pkl', './plmsearch_data/train/caths40/embedding.pkl'], help="Mean esm result.")
    parser.add_argument('-if', '--input_fasta', type=list, default=['./plmsearch_data/train/scope40_train/protein.fasta', './plmsearch_data/train/caths40/protein.fasta'], help="Iuput protein list, decide the protein order of the ss_mat.")
    parser.add_argument('-im', '--input_mat', type=list, default=['./plmsearch_data/train/scope40_train/ss_mat.npy', './plmsearch_data/train/caths40/ss_mat.npy'], help="Input mat name.")
    parser.add_argument('-pm', '--pretrained_model', default=None)
    parser.add_argument('-d', '--device-id', default=[0], nargs='*', help='gpu device list, if only cpu then set it None or empty')

    #output
    parser.add_argument('-smp', '--save_model_path', type=str, help='Pretrained MTPLM model for resuming training (optional)')

    # training parameters
    parser.add_argument('--ss_batch_size', type=int, default=100, help='minibatch size for ss loss (default: 100)')
    parser.add_argument('--epochs', type=int, default=200, help='number ot epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.000001, help='learning rate (default: 1e-6)')

    args = parser.parse_args()
    
    if args.pretrained_model == None:
        model = plmsearch(embed_dim = 1280)
        print("Train PLMSearch")
    else:
        model = plmsearch(embed_dim = 1280)
        model.load_pretrained(args.pretrained_model)
        print(f"Train PLMSearch from pretrained model: {args.pretrained_model}")

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
    ss_dataset = plmsearch_dataset()
    for index, _ in enumerate(args.esm_embedding):
        ss_dataset.load_dataset(args.esm_embedding[index], args.input_fasta[index], args.input_mat[index], device)
    ss_dataset.finalize()
    ss_train_size = int(0.9 * len(ss_dataset))
    ss_valid_size = len(ss_dataset) - ss_train_size
    print(f"Train size: {ss_train_size}")
    print(f"Valid size: {ss_valid_size}")
    ss_train_dataset, ss_valid_dataset = torch.utils.data.random_split(ss_dataset, [ss_train_size, ss_valid_size])
  
    # iterators for the ss data
    batch_size = args.ss_batch_size
    ss_train_dataloader = torch.utils.data.DataLoader(ss_train_dataset
                                                    , batch_size=batch_size
                                                    , shuffle=True
                                                    )
    ss_valid_dataloader = torch.utils.data.DataLoader(ss_valid_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

    params = [p for p in model_methods.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr = args.lr)

    ## train the model
    print('# training model', file=sys.stderr)

    best_spearman_r = 0
    if args.save_model_path is not None:
        eval_file_path = os.path.join(os.path.dirname(args.save_model_path), os.path.basename(args.save_model_path)[:-4] + '_eval.txt')

    with open(eval_file_path, 'w') as f:
        for t in range(args.epochs):
            logger.info(f"-------------------------------\nEpoch {t+1}")
            train_ss(model_methods, ss_train_dataloader, device, optimizer)
            spearman_r = eval_ss(model_methods, ss_valid_dataloader, device, f)
            
            if spearman_r > best_spearman_r:
                logger.info(f"-------------------------------\nSave Model at Epoch {t+1}")
                best_spearman_r = spearman_r
                if args.save_model_path is not None:
                    make_parent_dir(args.save_model_path)
                    torch.save(model_methods.state_dict(), args.save_model_path)

if __name__ == '__main__':
    main()
