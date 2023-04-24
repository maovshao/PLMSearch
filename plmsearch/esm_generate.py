# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
import time
import pickle
import re

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer

def esm_generate(model_location, fasta, mean_esm_result, nogpu):
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()
    if isinstance(model, MSATransformer):
        raise ValueError(
            "This script currently does not handle models with MSA input (MSA Transformer)."
        )
    
    if torch.cuda.is_available() and not nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(fasta)
    batches = dataset.get_batch_indices(16384, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches
    )
    print(f"Read {fasta} with {len(dataset)} sequences")

    mean_embedding_dic = {}
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and not nogpu:
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=[33], return_contacts=False)["representations"][33]

            for i, label in enumerate(labels):
                #get mean embedding
                label = re.split('\s+',label)[0]
                #mean_embedding_dic[label] = out[i, 1 : len(strs[i]) + 1].mean(0).clone().cpu()
                mean_embedding_dic[label] = out[i, 1 : len(strs[i]) + 1].mean(0).clone()
        
        with open(mean_esm_result, 'wb') as handle:
            pickle.dump(mean_embedding_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #input
    parser.add_argument('-emp', '--esm_model_path', type=str, default='./plmsearch/esm/saved_models/esm1b_t33_650M_UR50S.pt', help="ESM model location")
    parser.add_argument('-f', '--fasta', type=str, help="Fasta file to generate esm_embedding")
    #output
    parser.add_argument('-m', '--mean_esm_result', type=str, help="Esm result to use in cluster")
    #parameter
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    args = parser.parse_args()

    time_start=time.time()
    esm_generate(args.esm_model_path, args.fasta, args.mean_esm_result, args.nogpu)

    time_end=time.time()

    print('Esm embedding generate time cost:', time_end-time_start, 's')