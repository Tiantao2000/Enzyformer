from scripts.infer import *
from scripts.infer import infer_user_max_sep
from argparse import ArgumentParser
import sys
sys.path.append("./scripts")
import pandas as pd
from rdkit import Chem

def can_reaction(rxns):
    new_rxn = []
    for rxn in rxns:
        reac = Chem.MolToSmiles(Chem.MolFromSmiles(rxn.split(">>")[0]))
        prod = Chem.MolToSmiles(Chem.MolFromSmiles(rxn.split(">>")[-1]))
        can_rxn = reac+">>"+prod
        new_rxn.append(can_rxn)
    return new_rxn

def main(args):
    rxns = args['user_rxn']
    data = {
    "Entry": [f"id_user_{i}" for i in range(len(rxns))],
    "Sequence": rxns,
    "EC Number": ["unknown"] * len(rxns)}
    df = pd.DataFrame(data)

    rxns = list(df["Sequence"])
    new_rxns = can_reaction(rxns)
    df["sequence"] = new_rxns

    train_data = f"rxn_{args['rank']}"
    modelname = f"rxn_{args['rank']}_{args['train_mode']}"
    fingerprint = args['fingerprint']
    rxn_emb = pickle.load(
        open('./data/distance_map/' + train_data + f'_{args["fingerprint"]}_emb.pkl',
             'rb'))
    dim_input = rxn_emb.size()[1]
    infer_user_max_sep(train_data, df, fingerprint, dim_input, pretrained=False, model_name=modelname)





if __name__ == '__main__':
    parser = ArgumentParser('csv generation')
    parser.add_argument('-r', '--rank', default='first', help='rank to use')
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
