from inference_rxn_case import ECInferencer
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, cohen_kappa_score, matthews_corrcoef
import pickle

def is_valid_smiles(smi: str) -> bool:
    if not isinstance(smi, str):
        return False
    try:
        mol = Chem.MolFromSmiles(smi)
        return mol is not None
    except:
        return False

def main(args):
    df = pd.read_csv(
        "/home/tiantao/bioretro/Enzyformer/outputs/predicted_reactants_mode3_wo.csv",
        sep="\t"
    )

    raw_test_file = pd.read_csv(
        f"/home/tiantao/bioretro/CLEAN/data/test_rxn_{args['rank']}.csv"
    )

    assert len(df) == len(raw_test_file)

    
    ec_inferencer = ECInferencer(args)

    all_real_ecs, all_pred_ecs = [],[]
    all_raw_ecs = []

    for idx in tqdm(range(len(df))):
        row = df.iloc[idx]
        gt_row = raw_test_file.iloc[idx]

        ground_truth_reaction = gt_row["Sequence"]
        real_ec = gt_row["EC Number"]
        all_real_ecs.append(real_ec)

        target = row["target_smiles"]

        # sanity check
        gt_product = ground_truth_reaction.split(">>")[-1]


        # ---------- generated reactions ----------
        pred_reactions = []
        for j in range(1, 11):
            smi = row[f"sampled_smiles_{j}"]
            if pd.notna(smi) and is_valid_smiles(smi):
                pred_reactions.append(f"{smi}>>{target}")
        

        series_reactions = [ground_truth_reaction] + pred_reactions

        pred_ecs = ec_inferencer.infer(series_reactions)
        pred_ecs = list(pred_ecs)
        all_raw_ecs.append(pred_ecs)
        # for rxn in series_reactions:
        #     ec = ec_inferencer.infer([rxn])
        #     pred_ecs.append(ec)
        
        # 设置权重（线性递减）
        weights = [len(series_reactions) - i for i in range(len(series_reactions))]

        weighted_votes = Counter()
        for ec, w in zip(pred_ecs, weights):
            weighted_votes[ec] += w

        final_ec = weighted_votes.most_common(1)[0][0]

        print("Predicted EC number:", final_ec)
        all_pred_ecs.append(final_ec)
    # all_real_ecs: list of ground truth EC numbers
    # all_pred_ecs: list of predicted EC numbers
    
    with open(f"/home/tiantao/bioretro/CLEAN/results/new_eval/all_raw_ecs_{args['rank']}_mode_{args['mode']}.pkl", "wb") as f:
        pickle.dump(all_raw_ecs, f)


    true_label_flat = all_real_ecs
    true_label_flat = all_pred_ecs

    
    pre = precision_score(true_label_flat, pred_label_flat, average='weighted', zero_division=0)
    rec = recall_score(true_label_flat, pred_label_flat, average='weighted', zero_division=0)
    f1 = f1_score(true_label_flat, pred_label_flat, average='weighted', zero_division=0)
    acc = accuracy_score(true_label_flat, pred_label_flat)
    kappa = cohen_kappa_score(true_label_flat, pred_label_flat)
    mcc = matthews_corrcoef(true_label_flat, pred_label_flat)

    print("############ EC calling results using maximum separation ############")
    print('-' * 75)
    print(f'>>> total samples: {len(true_label_flat)} | total ec: {len(true_label_flat)} ')
    print(f'>>> precision: {pre:.3f} | recall: {rec:.3f} | F1: {f1:.3f} | ACC: {acc:.3f}')
    print(f'>>> Cohen\'s Kappa: {kappa:.3f} | MCC: {mcc:.3f}')
    print('-' * 75)

    # 写入 log 文件
    log_file = f"/home/tiantao/bioretro/CLEAN/results/new_eval/log_files_{args['rank']}_mode_{args['mode']}.txt"
    with open(log_file, 'a') as f:
        f.write("############ EC calling results using maximum separation ############\n")
        f.write('-'*75 + '\n')
        f.write(f'>>> total samples: {len(true_label_flat)} | total ec: {len(set(true_label_flat))}\n')
        f.write(f'>>> precision: {pre:.3f} | recall: {rec:.3f} | F1: {f1:.3f} | ACC: {acc:.3f}\n')
        f.write(f'>>> Cohen\'s Kappa: {kappa:.3f} | MCC: {mcc:.3f}\n')
        f.write('-'*75 + '\n')



if __name__ == '__main__':
    parser = ArgumentParser('csv generation')
    parser.add_argument('-rxn_file', '--rxn_file', default='/home/tiantao/bioretro/Enzyformer/outputs/final_results_mode_3.txt', help='rank to use')
    parser.add_argument('-ground_truth_file', '--ground_truth_file', default='/home/tiantao/bioretro/Data/test.targets.txt', help='rank to use')
    parser.add_argument('-mode', '--mode', default="first", help='rank to use')
    parser.add_argument('-r', '--rank', default='third', help='rank to use')
    parser.add_argument('-i', '--infer_mode', default='max', help='infer_mode')
    parser.add_argument('-t', '--train_mode', default='triplet', help='train_mode')
    parser.add_argument('-rxn', '--user_rxn', default=['O=C1NC(CN1C2=CC=C(O)C=C2)=O>>NC(NC(C1=CC=C(O)C=C1)C(O)=O)=O'], nargs='+')

    parser.add_argument(
        "-f", "--fingerprint",
        choices=["bert", "rxnfp", "drfp"],
        default="rxnfp",
        help="Fingerprint type to use"
    )


    args = parser.parse_args().__dict__
    main(args)
