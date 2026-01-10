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

    train_data_csv = pd.read_csv(f"./data/{train_data}.csv")
    id_ec_train, ec_id_dict_train = get_ec_id_dict_csv(train_data_csv)

    device = torch.device("cuda:0")
    dtype = torch.float32

    checkpoint = torch.load(f'./data/model/{fingerprint}/' + modelname + '.pth')
    model = LayerNormNet(512, 128, device, dtype, dim_input)
    model.load_state_dict(checkpoint)
    model.eval()
    emb_train = model(fp_embedding(ec_id_dict_train, fingerprint, device, dtype, train_data_csv))

    ec_number = infer_user_max_sep(train_data, df, fingerprint, emb_train, model)
    print(ec_numer)

class ECInferencer:
    def __init__(self, args):
        self.args = args
        self.rank = args["rank"]
        self.train_mode = args["train_mode"]
        self.fingerprint = args["fingerprint"]

        self.train_data = f"rxn_{self.rank}"
        self.modelname = f"{self.train_data}_{self.train_mode}"

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = torch.float32

        # ---------- load train embedding ----------
        rxn_emb = pickle.load(
            open(
                f"./data/distance_map/{self.train_data}_{self.fingerprint}_emb.pkl",
                "rb"
            )
        )
        self.dim_input = rxn_emb.size(1)

        # ---------- load train csv ----------
        self.train_data_csv = pd.read_csv(f"./data/{self.train_data}.csv")
        self.id_ec_train, self.ec_id_dict_train = get_ec_id_dict_csv(
            self.train_data_csv
        )

        # ---------- load model ----------
        checkpoint = torch.load(
            f"./data/model/{self.fingerprint}/{args['mode']}/{self.modelname}.pth",
            map_location=self.device
        )

        self.model = LayerNormNet(
            512, 128, self.device, self.dtype, self.dim_input
        ).to(self.device)

        self.model.load_state_dict(checkpoint)
        self.model.eval()

        # ---------- compute emb_train ONCE ----------
        with torch.no_grad():
            self.emb_train = self.model(
                fp_embedding(
                    self.ec_id_dict_train,
                    self.fingerprint,
                    self.device,
                    self.dtype,
                    self.train_data_csv
                )
            )

    def infer(self, user_rxns):
        """
        user_rxns: list[str]
        """
        df = pd.DataFrame({
            "Entry": [f"id_user_{i}" for i in range(len(user_rxns))],
            "Sequence": user_rxns,
            "EC Number": ["unknown"] * len(user_rxns)
        })

        # canonicalize
        df["Sequence"] = can_reaction(list(df["Sequence"]))

        with torch.no_grad():
            ec_number = infer_user_max_sep(
                self.train_data,
                df,
                self.fingerprint,
                self.emb_train,
                self.model
            )

        return ec_number




if __name__ == '__main__':
    parser = ArgumentParser('csv generation')
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
