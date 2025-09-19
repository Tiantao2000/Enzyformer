import hydra
import pandas as pd
from typing import List

import molbart.utils.data_utils as util
from molbart.models import Chemformer
from argparse import Namespace
from omegaconf import OmegaConf
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from molbart.utils.tokenizers import ChemformerTokenizer
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

def write_predictions(args, smiles, log_lhs, original_smiles):
    num_data = len(smiles)
    beam_width = len(smiles[0])
    beam_outputs = [[[]] * num_data for _ in range(beam_width)]
    beam_log_lhs = [[[]] * num_data for _ in range(beam_width)]

    for b_idx, (smiles_beams, log_lhs_beams) in enumerate(zip(smiles, log_lhs)):
        for beam_idx, (smi, log_lhs) in enumerate(zip(smiles_beams, log_lhs_beams)):
            beam_outputs[beam_idx][b_idx] = smi
            beam_log_lhs[beam_idx][b_idx] = log_lhs

    df_data = {"target_smiles": original_smiles}
    for beam_idx in range(beam_width):
        df_data["sampled_smiles_" + str(beam_idx + 1)] = beam_outputs[beam_idx]

    for beam_idx in range(beam_width):
        df_data["loglikelihood_" + str(beam_idx + 1)] = beam_log_lhs[beam_idx]

    df = pd.DataFrame(data=df_data)
    return df

def write_attentions(args, attns):
    try:
        with open("vis/attention.pkl", "wb") as f:
            pickle.dump(attns, f)
        print(f"Write attns to {args.attn_path}")
    except:
        "Failed to write attns"

def get_args():
    args_dict = {
        'batch_size': 64,
        'n_beams': 10,
        'n_unique_beams': None,
        'n_gpus': 1,
        'infile': '/home/tiantao/bioretro/bioformer/data/bioretro_without_classification/bioformer_test_tgt_wo.txt',
        'groundtruthfile': '/home/tiantao/bioretro/Data/test.targets.txt',
        'data_path': '/home/tiantao/bioretro/bioformer/data/bioretro_without_classification/bioformer_test_tgt_wo_sample.txt',
        'output_sampled_smiles': '/home/tiantao/bioretro/bioformer/outputs/dehfuehfu.csv',
        'vocabulary_path': 'bart_vocab_downstream.json',
        'task': 'backward_prediction',
        'i_chunk': 0,
        'n_chunks': 1,
        'data_device': 'cuda',
        'model_path': '/home/tiantao/bioretro/bioformer/chemformer_models/finetune/last_mode3.ckpt',
        'model_type': 'bart',
        'dataset_part': 'full',
        'train_mode': 'eval',
        'datamodule': ['SynthesisDataModule'],
        'select_index': 0,
        'mode_output':3
    }

    args = Namespace(**args_dict)
    args_dict = vars(args)  # Namespace → dict
    args = OmegaConf.create(args_dict)
    return args


def plot_all_layers_attention(attns_decoder_mha_batch, y_labels, x_labels, idx,
                              ncols=3, save_dir="/home/tiantao/bioretro/Chemformer/vis"):
    """
    绘制所有层的 cross-attention heatmap，拼接成一张总图

    Parameters
    ----------
    attns_decoder_mha_batch : list or tensor
        注意力矩阵，形状 [batch, n_layers, n_heads, seq_len, seq_len]
    output_tokens : list
        解码端 token 列表 (去掉首尾 <s> 和 </s>)
    input_tokens : list
        编码端 token 列表 (去掉首尾 <s> 和 </s>)
    idx : int
        当前样本编号，用于文件命名
    ncols : int, optional
        每行显示的子图数量，默认 4
    save_dir : str, optional
        保存路径
    """


    # 自定义配色
    cmap = LinearSegmentedColormap.from_list("light_red_to_dark_red", ["#ffe5e5", "darkred"])

    # 取 batch 中的第一个样本
    n_layers = len(attns_decoder_mha_batch)

    # 子图布局
    nrows = int(np.ceil(n_layers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    # 统一成一维数组
    axes = np.array(axes).flatten()

    for i in range(n_layers):
        # 取第 i 层的第一个 head
        attn = attns_decoder_mha_batch[i][0].detach().cpu().numpy()
        attn = attn[1:, 1:-1]  # 去掉 <s> 和 </s>

        ax = axes[i]
        sns.heatmap(
            attn,
            cmap=cmap,
            vmin=0, vmax=1,
            xticklabels=False,
            yticklabels=False,
            cbar=False,  # 全局只留一个 colorbar
            ax=ax
        )
        ax.invert_yaxis()
        ax.set_title(f"Layer {i}", fontsize=16, weight="bold")
        # ax.tick_params(axis='x', rotation=90, labelsize=12)
        # ax.tick_params(axis='y', rotation=0, labelsize=12)

        ax.set_xticks(np.arange(attn.shape[1]) + 0.5)  # heatmap默认刻度在格子中心
        ax.set_xticklabels([])
        ax.tick_params(axis='x', bottom=True, top=False, length=3, width=1.0)  # 刻度线保留
        ax.set_yticks(np.arange(attn.shape[0]) + 0.5)
        ax.set_yticklabels([])
        ax.tick_params(axis='y', left=True, right=False, length=3, width=1.0)

        # 边框黑色
        for side in ["top", "bottom", "left", "right"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.5)   # 边框加粗
            ax.spines[side].set_edgecolor("black")

    # 删除多余空子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # 调整子图，留出右边放色条
    fig.subplots_adjust(right=0.88)

    # 全局 colorbar
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(
        axes[0].collections[0],
        cax=cbar_ax
    )
    cbar.set_label("Attention", fontsize=16, weight="bold")
    cbar.ax.tick_params(labelsize=12)
    cbar.outline.set_linewidth(1.2)
    cbar.outline.set_edgecolor("black")

    plt.savefig(
        f"{save_dir}/cross_attention_all_layers_{idx}.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()
    print(f"Saved at {save_dir}/cross_attention_all_layers_{idx}.png")



def main(args):
    # chemformer = Chemformer(args)
    # print("Making predictions...")
    # smiles, log_lhs, original_smiles = chemformer.predict(dataset=args.dataset_part)
    # df = write_predictions(args, smiles, log_lhs, original_smiles)
    df = pd.read_csv(f"/home/tiantao/bioretro/bioformer/outputs/predicted_reactants_mode{args.mode_output}_wo.csv", sep="\t")
    pred_reactants_list = list(df["sampled_smiles_1"])
    # for idx in tqdm(range(len(pred_reactants_list))):

    for idx in tqdm(range(len(pred_reactants_list))):
        with open(args.infile, "r") as f:
            lines = [line.strip() for line in f if line.strip()]  ##产物
        with open(args.groundtruthfile, "r") as f:
            groungtruthlines = [line.strip().replace(" ", "") for line in f if line.strip()]  ##反应物
        pred_reaction_list, real_reaction_list = [],[]
        for reac, prod in zip(pred_reactants_list, lines):
            pred_reaction_list.append((reac+">>"+prod))
        for reac, prod in zip(groungtruthlines, lines):
            real_reaction_list.append((reac+">>"+prod))
        # 按长度排序，取前 n 行
        # shortest = sorted(reaction_list, key=len)[args.select_index]
        select = pred_reaction_list[idx]
        print(f"pred_reaction is {select}, real_raction is {real_reaction_list[idx]}")
        with open(args.data_path, "w") as f:
            f.write(select)   ##直接写入反应

        pred_reactant = pred_reactants_list[idx]
        try:
            pred_reactant_can = Chem.MolToSmiles(Chem.MolFromSmiles(pred_reactant))
        except:
            print(f"error smiles for {pred_reactant}")
            continue

        target_reactant_can = Chem.MolToSmiles(Chem.MolFromSmiles(real_reaction_list[idx].split('>>')[0]))

        # 如果不相等，抛出错误并显示信息
        if pred_reactant_can != target_reactant_can:
            print(f"Prediction does not match target! pred: {pred_reactant_can}, target: {target_reactant_can}")
            continue
        print("success matching")

        chemformer = Chemformer(args)
        for i, batch in enumerate(chemformer.datamodule.full_dataloader()):

            batch = chemformer.on_device(batch)
            (
                model_input_batch,
                model_output_batch,
                attns_encoder_batch,
                attns_decoder_batch,
                attns_decoder_mha_batch,
            ) = chemformer.model(batch,get_attn=True)
            # 初始化 tokenizer
            tokenizer = ChemformerTokenizer(filename="bart_vocab_downstream.json")
            input_tokens = tokenizer.tokenize([select.split('>>')[-1]]
                )[0]

            output_tokens = tokenizer.tokenize([select.split('>>')[0]]
                )[0]

        plt.figure(figsize=(10, 8))  # 单张图大小
        layer_attn_0 = attns_decoder_mha_batch[0][0].detach().cpu().numpy()
        layer_attn_0 = layer_attn_0[1:, 1:-1]

        y_labels = output_tokens[1:-1]
        x_labels = input_tokens[1:-1]
        plot_all_layers_attention(attns_decoder_mha_batch,y_labels,x_labels,idx)
        cmap = LinearSegmentedColormap.from_list("light_red_to_dark_red", ["#ffe5e5", "darkred"])

        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(
            layer_attn_0,
            cmap=cmap,
            vmin=0, vmax=1,  # 固定范围 [0,1]
            xticklabels=False,
            yticklabels=False,
            cbar_kws={"shrink": 0.6}
        )
        ax.invert_yaxis()

        ax.set_title("Layer 0", fontsize=18, weight="bold")
        ax.set_xticks(np.arange(layer_attn_0.shape[1]) + 0.5)  # heatmap默认刻度在格子中心
        ax.set_xticklabels([])
        ax.tick_params(axis='x', bottom=True, top=False, length=3, width=1.0)  # 刻度线保留
        ax.set_yticks(np.arange(layer_attn_0.shape[0]) + 0.5)
        ax.set_yticklabels([])
        ax.tick_params(axis='y', left=True, right=False, length=3, width=1.0)


        # Colorbar 字体设置
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.set_ylabel("Attention", fontsize=16)

        # 边框黑色
        for side in ["top", "bottom", "left", "right"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.5)   # 边框加粗
            ax.spines[side].set_edgecolor("black")

        plt.tight_layout()
        plt.savefig(
            f'/home/tiantao/bioretro/Chemformer/vis/cross_attention_heatmap_layer0_{idx}.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.show()
        print("Finished Extraction.")
        print("Output attentions")






if __name__ == "__main__":
    args = get_args()
    main(args)
