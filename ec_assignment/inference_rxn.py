from scripts.infer import *
from argparse import ArgumentParser
import sys
sys.path.append("/home/tiantao/bioretro/CLEAN/scripts")
import pandas as pd

def main(args):
    csv_file = pd.read_csv(args["dataset"])
    filtered_df = csv_file[csv_file["split"].isin(["test"])].reset_index(drop=True)
    ecs = list(filtered_df[f"{args['rank']}_rank_ec"])
    rxns = list(filtered_df["EnzymaticReaction"])
    entryid = [f"id_{i}" for i in range(len(filtered_df))]
    df = pd.DataFrame({
        'Entry': entryid,
        'EC Number': ecs,
        'Sequence': rxns
    })
    df.to_csv(f"./data/test_rxn_{args['rank']}.csv", index=False)

    train_data = f"rxn_{args['rank']}"
    modelname = f"rxn_{args['rank']}_{args['train_mode']}"
    fingerprint = args['fingerprint']
    rxn_emb = pickle.load(
        open('./data/distance_map/' + train_data + f'_{args["fingerprint"]}_emb.pkl',
             'rb'))
    dim_input = rxn_emb.size()[1]
    infer_maxsep(train_data, df, fingerprint, dim_input, report_metrics=True, pretrained=False, model_name=modelname)






if __name__ == '__main__':
    parser = ArgumentParser('csv generation')
    parser.add_argument('-d', '--dataset', default='../Data/enzymatic_reactions.csv', help='Dataset to use')
    parser.add_argument('-r', '--rank', default='first', help='rank to use')
    parser.add_argument('-i', '--infer_mode', default='max', help='infer_mode')
    parser.add_argument('-t', '--train_mode', default='triplet', help='train_mode')
    parser.add_argument(
        "-f", "--fingerprint",
        choices=["bert", "rxnfp", "drfp"],
        default="rxnfp",
        help="Fingerprint type to use"
    )
    args = parser.parse_args().__dict__
    main(args)