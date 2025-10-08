import torch
import time
import os
import pickle
from scripts.model import *
from scripts.utils import *
import torch.nn as nn
import argparse
from scripts.distance_map import get_dist_map
import pandas as pd
from scripts.dataloader import *
from tqdm import tqdm

def get_ec_id_dict_for_csv(df) -> dict:
    id_ec = df.set_index('Entry')['EC Number'].str.split(';').to_dict()

    # 创建ec_id字典（使用pandas的groupby优化）
    ec_id = {}
    exploded = df.assign(ec_number=df['EC Number'].str.split(';')).explode('EC Number')

    for ec, group in exploded.groupby('EC Number'):
        ec_id[ec] = set(group['Entry'])

    return id_ec, ec_id

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-e', '--epoch', type=int, default=50000)
    parser.add_argument('-n', '--model_name', type=str, default='rxn_triplet')
    parser.add_argument('-t', '--training_data', type=str, default='rxn')
    parser.add_argument('-d', '--hidden_dim', type=int, default=512)
    parser.add_argument('-o', '--out_dim', type=int, default=128)
    parser.add_argument(
        "-f", "--fingerprint",
        choices=["bert", "rxnfp", "drfp"],
        default="drfp",
        help="Fingerprint type to use"
    )
    parser.add_argument('--adaptive_rate', type=int, default=100)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()
    return args


def get_dataloader(dist_map, id_ec, ec_id, args):
    params = {
        'batch_size': 6000,
        'shuffle': True,
    }
    negative = mine_hard_negative(dist_map, 30)
    train_data = Triplet_dataset_with_mine_EC(id_ec, ec_id, negative, args)
    train_loader = torch.utils.data.DataLoader(train_data, **params)
    return train_loader


def train(model, args, epoch, train_loader,
          optimizer, device, dtype, criterion):
    model.train()
    total_loss = 0.
    start_time = time.time()

    for batch, data in enumerate(train_loader):
        optimizer.zero_grad()
        anchor, positive, negative = data
        anchor_out = model(anchor.to(device=device, dtype=dtype))
        positive_out = model(positive.to(device=device, dtype=dtype))
        negative_out = model(negative.to(device=device, dtype=dtype))

        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if args.verbose:
            lr = args.learning_rate
            ms_per_batch = (time.time() - start_time) * 1000
            cur_loss = total_loss 
            print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_loader):5d} batches | '
                  f'lr {lr:02.4f} | ms/batch {ms_per_batch:6.4f} | '
                  f'loss {cur_loss:5.2f}')
            start_time = time.time()
    # record running average training loss
    return total_loss/(batch + 1)


def main():
    seed_everything()
    ensure_dirs('./data/model')
    args = parse()
    torch.backends.cudnn.benchmark = True
    df = pd.read_csv('./data/' + args.training_data + '.csv')
    id_ec, ec_id_dict = get_ec_id_dict_for_csv(df)
    ec_id = {key: list(ec_id_dict[key]) for key in ec_id_dict.keys()}
    #======================== override args ====================#
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    lr, epochs = args.learning_rate, args.epoch
    model_name = args.model_name
    print('==> device used:', device, '| dtype used: ',
          dtype, "\n==> args:", args)
    #======================== ESM embedding  ===================#
    # loading ESM embedding for dist map

    rxn_emb = pickle.load(
        open('./data/distance_map/' + args.training_data + f'_{args.fingerprint}_emb.pkl',
                'rb')).to(device=device, dtype=dtype)
    dist_map = pickle.load(open('./data/distance_map/' + \
        args.training_data + f'_{args.fingerprint}.pkl', 'rb'))
    #======================== initialize model =================#
    dim_input = rxn_emb.size()[1]
    model = LayerNormNet(args.hidden_dim, args.out_dim, device, dtype, dim_input)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.TripletMarginLoss(margin=1, reduction='mean')
    best_loss = float('inf')
    train_loader = get_dataloader(dist_map, id_ec, ec_id, args)
    print("The number of unique EC numbers: ", len(dist_map.keys()))
    #======================== training =======-=================#
    # training
    for epoch in tqdm(range(1, epochs + 1)):
        if epoch % args.adaptive_rate == 0 and epoch != epochs + 1:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, betas=(0.9, 0.999))
            # save updated model
            torch.save(model.state_dict(), f'./data/model/{args.fingerprint}/' +
                       model_name + '_' + str(epoch) + '.pth')
            # delete last model checkpoint
            if epoch != args.adaptive_rate:
                os.remove(f'./data/model/{args.fingerprint}/' + model_name + '_' +
                          str(epoch-args.adaptive_rate) + '.pth')
            # sample new distance map
            dist_map = get_dist_map(
                ec_id_dict, rxn_emb, device, dtype, model=model)
            train_loader = get_dataloader(dist_map, id_ec, ec_id, args)
        # -------------------------------------------------------------------- #
        epoch_start_time = time.time()
        train_loss = train(model, args, epoch, train_loader,
                           optimizer, device, dtype, criterion)
        # only save the current best model near the end of training
        if (train_loss < best_loss and epoch > 0.8*epochs):
            torch.save(model.state_dict(), f'./data/model/{args.fingerprint}/' + model_name + '.pth')
            best_loss = train_loss
            print(f'Best from epoch : {epoch:3d}; loss: {train_loss:6.4f}')

        elapsed = time.time() - epoch_start_time
        print('-' * 75)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'training loss {train_loss:6.4f}')
        print('-' * 75)
    # remove tmp save weights
    os.remove(f'./data/model/{args.fingerprint}/' + model_name + '.pth')
    os.remove(f'./data/model/{args.fingerprint}/' + model_name + '_' + str(epoch) + '.pth')
    # save final weights
    torch.save(model.state_dict(), f'./data/model/{args.fingerprint}/' + model_name + '.pth')


if __name__ == '__main__':
    main()
