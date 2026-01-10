import csv
import random
import os
from re import L
import torch
import numpy as np
import subprocess
import pickle
import sys
from argparse import ArgumentParser
import sys
sys.path.append("./scripts")
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from distance_map import get_dist_map
# from rxnfp.models import SmilesClassificationModel
from rdkit.Chem import AllChem
import pandas as pd
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)
model, tokenizer = get_default_model_and_tokenizer()
rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
from tqdm import tqdm
from drfp import DrfpEncoder

def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_bert(id, model, csv_file):
    rxn = list(csv_file[csv_file['Entry'] == id]['Sequence'])[0]
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "/home/tiantao/bioretro/SynthCoder/pretraining_logs/TensorBoard_logs/version_2/checkpoints/last.ckpt.dir/")

    inputs = tokenizer(rxn, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.distilbert(**inputs)  # 只取 encoder
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        cls_embedding = last_hidden_state[:, 0, :]  # [CLS] token embedding

    embedding = last_hidden_state.mean(dim=1)  # [batch, hidden_dim]
    fp_float32 = embedding.float()
    torch.save(fp_float32, f"./data/bert/{id}.pt")
    return fp_float32


def load_rxn(id, csv_file):
    rxn = list(csv_file[csv_file['Entry'] == id]['Sequence'])[0]
    fp = rxnfp_generator.convert(rxn)
    fp_float32 = torch.tensor(fp, dtype=torch.float32).reshape(1, len(fp))
    torch.save(fp_float32, f"./data/rxnfp/{id}.pt")
    return fp_float32

def load_drfp(id, csv_file):
    rxn = list(csv_file[csv_file['Entry'] == id]['Sequence'])[0]
    fp = DrfpEncoder.encode(rxn)
    fp_float32 = torch.tensor(fp[0], dtype=torch.float32).unsqueeze(0)
    torch.save(fp_float32, f"./data/drfp/{id}.pt")
    return fp_float32


def fp_embedding(ec_id_dict, fp, device, dtype, train_file):
    '''
    Loading esm embedding in the sequence of EC numbers
    prepare for calculating cluster center by EC
    '''
    rxnfp_emb = []
    # for ec in tqdm(list(ec_id_dict.keys())):
    bert_model = DistilBertForSequenceClassification.from_pretrained(
        "/home/tiantao/bioretro/SynthCoder/pretraining_logs/TensorBoard_logs/version_2/checkpoints/last.ckpt.dir/"
    )
    for ec in tqdm(list(ec_id_dict.keys())):
        ids_for_query = list(ec_id_dict[ec])
        if fp == "bert":
            esm_to_cat = [load_bert(id, bert_model, train_file) for id in ids_for_query]
        elif fp == "rxnfp":
            esm_to_cat = [load_rxn(id, train_file) for id in ids_for_query]
        elif fp == "drfp":
            esm_to_cat = [load_drfp(id, train_file) for id in ids_for_query]
        else:
            raise ValueError(f"Unknown fp type: {fp}")
        rxnfp_emb = rxnfp_emb + esm_to_cat
    return torch.cat(rxnfp_emb).to(device=device, dtype=dtype)


def get_ec_id_dict(df) -> dict:
    id_ec = df.set_index('Entry')['EC Number'].str.split(';').to_dict()

    # 创建ec_id字典（使用pandas的groupby优化）
    ec_id = {}
    exploded = df.assign(ec_number=df['EC Number'].str.split(';')).explode('EC Number')

    for ec, group in exploded.groupby('EC Number'):
        ec_id[ec] = set(group['Entry'])

    return id_ec, ec_id

def compute_rxnfp_distance(train_file, fp, name = "rxnfp"):
    ensure_dirs('./data/distance_map/')
    _, ec_id_dict = get_ec_id_dict(train_file)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    rxn_emb = fp_embedding(ec_id_dict, fp, device, dtype, train_file)
    rxn_dist = get_dist_map(ec_id_dict, rxn_emb, device, dtype)
    pickle.dump(rxn_dist, open('./data/distance_map/' + name + f'_{fp}.pkl', 'wb'))
    pickle.dump(rxn_emb, open('./data/distance_map/' + name + f'_{fp}_emb.pkl', 'wb'))


def main(args):
    os.makedirs(f"./data/{args['fingerprint']}", exist_ok=True)
    csv_file = pd.read_csv(args["dataset"])
    filtered_df = csv_file[csv_file["split"].isin(["train", "valid"])].reset_index(drop=True)
    ecs = list(filtered_df[f"{args['rank']}_rank_ec"])
    rxns = list(filtered_df["EnzymaticReaction"])
    entryid = [f"id_{i}" for i in range(len(filtered_df))]
    df = pd.DataFrame({
        'Entry': entryid,
        'EC Number': ecs,
        'Sequence': rxns
    })
    df.to_csv(f"./data/rxn_{args['rank']}.csv", index=False)
    fp = args["fingerprint"]
    compute_rxnfp_distance(df, fp, name = f"rxn_{args['rank']}")







if __name__ == '__main__':
    parser = ArgumentParser('csv generation')
    parser.add_argument('-d', '--dataset', default='./data/enzymatic_reactions.csv', help='Dataset to use')
    parser.add_argument('-r', '--rank', default='first', help='rank to use')
    parser.add_argument(
        "-f", "--fingerprint",
        choices=["bert", "rxnfp", "drfp"],
        default="drfp",
        help="Fingerprint type to use"
    )
    args = parser.parse_args().__dict__
    main(args)
