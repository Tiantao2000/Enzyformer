import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


class FuncLR(LambdaLR):
    def get_lr(self):
        return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]


# Use Pytorch implementation but with 'pre-norm' style layer normalisation
class PreNormEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, get_attn=False):
        # Self attention block
        att = self.norm1(src)
        att_out, attn_weights = self.self_attn(
            att, att, att,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True  # 新增
        )
        att = src + self.dropout1(att_out)

        # Feedforward block
        out = self.norm2(att)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = att + self.dropout2(out)
        if get_attn:
            return out, attn_weights
        else:
            return out


# Use Pytorch implementation but with 'pre-norm' style layer normalisation
class PreNormDecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None, get_attn=False
    ):
        # Self attention block
        query = self.norm1(tgt)
        self_attn_out, self_attn_weights = self.self_attn(
            query, query, query,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=True  # 新增
        )
        # query = self.self_attn(
        #     query,
        #     query,
        #     query,
        #     attn_mask=tgt_mask,
        #     key_padding_mask=tgt_key_padding_mask,
        # )[0]
        query = tgt + self.dropout1(self_attn_out)

        # Context attention block
        att = self.norm2(query)
        att, cross_attn_weights = self.multihead_attn(
            self.norm2(query), memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True  # 新增
        )
        att = query + self.dropout2(att)

        # Feedforward block
        out = self.norm3(att)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = att + self.dropout3(out)
        if get_attn:
            return out, self_attn_weights, cross_attn_weights
        else:
            return out
