import csv
import random
import os
from re import L
import torch
import numpy as np
import subprocess
import pickle
import sys
sys.path.append("/home/tiantao/bioretro/CLEAN/scripts")

from distance_map import get_dist_map


from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)
model, tokenizer = get_default_model_and_tokenizer()
rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
from tqdm import tqdm
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from drfp import DrfpEncoder

def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_ec_id_dict(csv_name: str) -> dict:
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    id_ec = {}
    ec_id = {}

    for i, rows in enumerate(csvreader):
        if i > 0:
            id_ec[rows[0]] = rows[1].split(';')
            for ec in rows[1].split(';'):
                if ec not in ec_id.keys():
                    ec_id[ec] = set()
                    ec_id[ec].add(rows[0])
                else:
                    ec_id[ec].add(rows[0])
    return id_ec, ec_id

def get_ec_id_dict_csv(df) -> dict:
    id_ec = df.set_index('Entry')['EC Number'].str.split(';').to_dict()

    # 创建ec_id字典（使用pandas的groupby优化）
    ec_id = {}
    exploded = df.assign(ec_number=df['EC Number'].str.split(';')).explode('EC Number')

    for ec, group in exploded.groupby('EC Number'):
        ec_id[ec] = set(group['Entry'])

    return id_ec, ec_id




def get_ec_id_dict_non_prom(csv_name: str) -> dict:
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    id_ec = {}
    ec_id = {}

    for i, rows in enumerate(csvreader):
        if i > 0:
            if len(rows[1].split(';')) == 1:
                id_ec[rows[0]] = rows[1].split(';')
                for ec in rows[1].split(';'):
                    if ec not in ec_id.keys():
                        ec_id[ec] = set()
                        ec_id[ec].add(rows[0])
                    else:
                        ec_id[ec].add(rows[0])
    return id_ec, ec_id


def format_esm(a):
    if type(a) == dict:
        a = a['mean_representations'][33]
    return a


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







def model_embedding_fp_test(id_ec_test, fp, model, device, dtype, test_file):
    ids_for_query = list(id_ec_test.keys())
    # for ec in tqdm(list(ec_id_dict.keys())):
    bert_model = DistilBertForSequenceClassification.from_pretrained(
        "/home/tiantao/bioretro/SynthCoder/pretraining_logs/TensorBoard_logs/version_2/checkpoints/last.ckpt.dir/"
    )
    if fp == "bert":
        esm_to_cat = [load_bert(id, bert_model, test_file) for id in ids_for_query]
    elif fp == "rxnfp":
        esm_to_cat = [load_rxn(id, test_file) for id in ids_for_query]
    elif fp == "drfp":
        esm_to_cat = [load_drfp(id, test_file) for id in ids_for_query]
    else:
        raise ValueError(f"Unknown fp type: {fp}")
    esm_emb = torch.cat(esm_to_cat).to(device=device, dtype=dtype)
    model_emb = model(esm_emb)
    return model_emb







def model_embedding_test_ensemble(id_ec_test, device, dtype):
    '''
    Instead of loading esm embedding in the sequence of EC numbers
    the test embedding is loaded in the sequence of queries
    '''
    ids_for_query = list(id_ec_test.keys())
    esm_to_cat = [load_esm(id) for id in ids_for_query]
    esm_emb = torch.cat(esm_to_cat).to(device=device, dtype=dtype)
    return esm_emb

def csv_to_fasta(csv_name, fasta_name):
    csvfile = open(csv_name, 'r')
    csvreader = csv.reader(csvfile, delimiter='\t')
    outfile = open(fasta_name, 'w')
    for i, rows in enumerate(csvreader):
        if i > 0:
            outfile.write('>' + rows[0] + '\n')
            outfile.write(rows[2] + '\n')
            
def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def retrive_esm1b_embedding(fasta_name):
    esm_script = "esm/scripts/extract.py"
    esm_out = "data/esm_data"
    esm_type = "esm1b_t33_650M_UR50S"
    fasta_name = "data/" + fasta_name + ".fasta"
    command = ["python", esm_script, esm_type, 
              fasta_name, esm_out, "--include", "mean"]
    subprocess.run(command)
 
def compute_esm_distance(train_file):
    ensure_dirs('./data/distance_map/')
    _, ec_id_dict = get_ec_id_dict('./data/' + train_file + '.csv')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    esm_emb = esm_embedding(ec_id_dict, device, dtype)
    esm_dist = get_dist_map(ec_id_dict, esm_emb, device, dtype)
    pickle.dump(esm_dist, open('./distance_map/' + train_file + '.pkl', 'wb'))
    pickle.dump(esm_emb, open('./distance_map/' + train_file + '_esm.pkl', 'wb'))
    
def prepare_infer_fasta(fasta_name):
    retrive_esm1b_embedding(fasta_name)
    csvfile = open('./data/' + fasta_name +'.csv', 'w', newline='')
    csvwriter = csv.writer(csvfile, delimiter = '\t')
    csvwriter.writerow(['Entry', 'EC number', 'Sequence'])
    fastafile = open('./data/' + fasta_name +'.fasta', 'r')
    for i in fastafile.readlines():
        if i[0] == '>':
            csvwriter.writerow([i.strip()[1:], ' ', ' '])
    
def main():
    compute_esm_distance("split10")

if __name__ == '__main__':
    main()