import torch
from .utils import * 
from .model import LayerNormNet
from .distance_map import *
from .evaluate import *
import pandas as pd
import warnings
import pandas as pd
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, accuracy_score
import numpy as np


def warn(*args, **kwargs):
    pass
warnings.warn = warn



def infer_pvalue(train_data, test_data, p_value = 1e-5, nk_random = 20, 
                 report_metrics = False, pretrained=True, model_name=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/' + train_data + '.csv')
    id_ec_test, _ = get_ec_id_dict('./data/' + test_data + '.csv')
    # load checkpoints
    # NOTE: change this to LayerNormNet(512, 256, device, dtype) 
    # and rebuild with [python build.py install]
    # if inferencing on model trained with supconH loss
    model = LayerNormNet(512, 128, device, dtype)
    
    if pretrained:
        try:
            checkpoint = torch.load('./data/pretrained/'+ train_data +'.pth')
        except FileNotFoundError as error:
            raise Exception('No pretrained weights for this training data')
    else:
        try:
            checkpoint = torch.load('./data/model/'+ model_name +'.pth')
        except FileNotFoundError as error:
            raise Exception('No model found!')
        
    model.load_state_dict(checkpoint)
    model.eval()
    # load precomputed EC cluster center embeddings if possible
    if train_data == "split70":
        emb_train = torch.load('./data/pretrained/70.pt')
    elif train_data == "split100":
        emb_train = torch.load('./data/pretrained/100.pt')
    else:
        emb_train = model(esm_embedding(ec_id_dict_train, device, dtype))
        
    emb_test = model_embedding_test(id_ec_test, model, device, dtype)
    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    seed_everything()
    eval_df = pd.DataFrame.from_dict(eval_dist)
    rand_nk_ids, rand_nk_emb_train = random_nk_model(
        id_ec_train, ec_id_dict_train, emb_train, n=nk_random, weighted=True)
    random_nk_dist_map = get_random_nk_dist_map(
        emb_train, rand_nk_emb_train, ec_id_dict_train, rand_nk_ids, device, dtype)
    ensure_dirs("./results")
    out_filename = "results/" +  test_data
    write_pvalue_choices( eval_df, out_filename, random_nk_dist_map, p_value=p_value)
    # optionally report prediction precision/recall/...
    if report_metrics:
        pred_label = get_pred_labels(out_filename, pred_type='_pvalue')
        pred_probs = get_pred_probs(out_filename, pred_type='_pvalue')
        true_label, all_label = get_true_labels('./data/' + test_data)
        pre, rec, f1, roc, acc = get_eval_metrics(
            pred_label, pred_probs, true_label, all_label)
        print(f'############ EC calling results using random '
        f'chosen {nk_random}k samples ############')
        print('-' * 75)
        print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
            f'>>> precision: {pre:.3} | recall: {rec:.3}'
            f'| F1: {f1:.3} | AUC: {roc:.3} ')
        print('-' * 75)  


def infer_maxsep(train_data, test_data, fp, dim_input, report_metrics = False,
                 pretrained=True, model_name=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    train_data_csv = pd.read_csv(f"./data/{train_data}.csv")
    id_ec_train, ec_id_dict_train = get_ec_id_dict_csv(train_data_csv)
    id_ec_test, _ = get_ec_id_dict_csv(test_data)
    # load checkpoints
    # NOTE: change this to LayerNormNet(512, 256, device, dtype) 
    # and rebuild with [python build.py install]
    # if inferencing on model trained with supconH loss

    model = LayerNormNet(512, 128, device, dtype, dim_input)
    
    if pretrained:
        try:
            checkpoint = torch.load(f'./data/pretrained/{fp}/'+ train_data +'.pth', map_location=device)
        except FileNotFoundError as error:
            raise Exception('No pretrained weights for this training data')
    else:
        try:
            checkpoint = torch.load(f'./data/model/{fp}/'+ model_name +'.pth')
        except FileNotFoundError as error:
            raise Exception('No model found!')
            
    model.load_state_dict(checkpoint)
    model.eval()
    # load precomputed EC cluster center embeddings if possible
    if train_data == "split70":
        emb_train = torch.load('./data/pretrained/70.pt', map_location=device)
    elif train_data == "split100":
        emb_train = torch.load('./data/pretrained/100.pt', map_location=device)
    else:
        emb_train = model(fp_embedding(ec_id_dict_train,fp, device, dtype,train_data_csv))
        torch.save(emb_train, f"./data/pretrained/{fp}/{train_data}.pt")

    emb_test = model_embedding_fp_test(id_ec_test, fp, model, device, dtype, test_data)
    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    seed_everything()
    eval_df = pd.DataFrame.from_dict(eval_dist)
    ensure_dirs(f"./results/{fp}")
    out_filename = f"results/{fp}/" +  "text_" + train_data + ".csv"
    write_max_sep_choices(eval_df, out_filename)
    if report_metrics:
        pred_label = get_pred_labels(out_filename, pred_type='_maxsep')
        pred_probs = get_pred_probs(out_filename, pred_type='_maxsep')
        true_label, all_label = get_true_labels_csv(test_data)
        pre, rec, f1, roc, acc = get_eval_metrics(
            pred_label, pred_probs, true_label, all_label)
        true_label_flat = [x[0] for x in true_label]
        pred_label_flat = [x[0] for x in pred_label]
        kappa = cohen_kappa_score(true_label_flat, pred_label_flat)
        mcc = matthews_corrcoef(true_label_flat, pred_label_flat)
        print("############ EC calling results using maximum separation ############")
        print('-' * 75)
        print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
            f'>>> precision: {pre:.3} | recall: {rec:.3}'
            f'| F1: {f1:.3} | AUC: {roc:.3} | ACC: {acc:.3} ')
        print(f'>>> Cohen\'s Kappa: {kappa:.3} | MCC: {mcc:.3}')
        print('-' * 75)
        log_file = f"results/{fp}/log_files_{train_data}.txt"

        with open(log_file, 'a') as f:  # 使用 'a' 模式，追加写入
            msg = "############ EC calling results using maximum separation ############"
            f.write(msg + '\n')

            msg = '-' * 75
            f.write(msg + '\n')

            msg = f'>>> total samples: {len(true_label)} | total ec: {len(all_label)}'
            f.write(msg + '\n')

            msg = f'>>> precision: {pre:.3f} | recall: {rec:.3f} | F1: {f1:.3f} | AUC: {roc:.3f} | ACC: {acc:.3f}'
            f.write(msg + '\n')

            msg = f'>>> Cohen\'s Kappa: {kappa:.3f} | MCC: {mcc:.3f}'
            f.write(msg + '\n')

            msg = '-' * 75
            f.write(msg + '\n')

def infer_user_max_sep(train_data, test_data, fp, dim_input,
                 pretrained=True, model_name=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    train_data_csv = pd.read_csv(f"./data/{train_data}.csv")
    id_ec_train, ec_id_dict_train = get_ec_id_dict_csv(train_data_csv)
    id_ec_test, _ = get_ec_id_dict_csv(test_data)
    # load checkpoints
    # NOTE: change this to LayerNormNet(512, 256, device, dtype)
    # and rebuild with [python build.py install]
    # if inferencing on model trained with supconH loss

    model = LayerNormNet(512, 128, device, dtype, dim_input)

    if pretrained:
        try:
            checkpoint = torch.load(f'./data/pretrained/{fp}/' + train_data + '.pth', map_location=device)
        except FileNotFoundError as error:
            raise Exception('No pretrained weights for this training data')
    else:
        try:
            checkpoint = torch.load(f'./data/model/{fp}/' + model_name + '.pth')
        except FileNotFoundError as error:
            raise Exception('No model found!')

    model.load_state_dict(checkpoint)
    model.eval()
    # load precomputed EC cluster center embeddings if possible
    if train_data == "split70":
        emb_train = torch.load('./data/pretrained/70.pt', map_location=device)
    elif train_data == "split100":
        emb_train = torch.load('./data/pretrained/100.pt', map_location=device)
    else:
        emb_train = model(fp_embedding(ec_id_dict_train, fp, device, dtype, train_data_csv))
        torch.save(emb_train, f"./data/pretrained/{fp}/{train_data}.pt")

    emb_test = model_embedding_fp_test(id_ec_test, fp, model, device, dtype, test_data)
    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    seed_everything()
    eval_df = pd.DataFrame.from_dict(eval_dist)
    min_classes = eval_df.idxmin(axis=0)

    print(min_classes)
    return min_classes

