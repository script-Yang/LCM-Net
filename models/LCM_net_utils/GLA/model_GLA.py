import os
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from models.model_utils import *

def init_non_llm(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if getattr(m, "in_proj_weight", None) is not None:
            nn.init.xavier_uniform_(m.in_proj_weight)
        if getattr(m, "in_proj_bias", None) is not None:
            nn.init.zeros_(m.in_proj_bias)
        if hasattr(m, "out_proj"):
            nn.init.xavier_uniform_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                nn.init.zeros_(m.out_proj.bias)

class TransformerBlock(nn.Module):
    def __init__(self, dim, nhead=8, mlp_ratio=4.0, p_drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=p_drop, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(p_drop),
        )
        self.ln2 = nn.LayerNorm(dim)
        self.apply(init_non_llm)

    def forward(self, x, attn_mask=None):
        # x: (B,L,D)
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        h = self.ln2(x)
        h = self.ffn(h)
        x = x + h
        return x

class RegisterSNN(nn.Module):
    def __init__(
        self,
        omic_input_dim: int,          # G
        model_size_omic: str = "small",
        k_register: int = 8,
        nhead: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.k_register = k_register
        dim_dict = {"small": 256}
        self.base_dim = dim_dict[model_size_omic]

        blocks = [SNN_Block(omic_input_dim, self.base_dim)]
        for _ in range(1):
            blocks.append(SNN_Block(self.base_dim, self.base_dim, dropout=0.25))
        self.fc_omic = nn.Sequential(*blocks)
        self.gene_proj = nn.Linear(1, self.base_dim)

        self.register = nn.Parameter(torch.randn(1, k_register, self.base_dim) * 0.02)

        self.ln = nn.LayerNorm(self.base_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.base_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.apply(init_non_llm)

    def forward(self, x, return_heatmap: bool = False):
        # need_w = return_heatmap
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B, G = x.shape
        h_global = self.fc_omic(x)                            
        gene_tokens = self.gene_proj(x.unsqueeze(-1))         
        gene_tokens = gene_tokens + h_global.unsqueeze(1)     

        reg_tokens = self.register.expand(B, -1, -1)          
        tokens = torch.cat([gene_tokens, reg_tokens], dim=1)  
        h = self.ln(tokens)
        # out, attn_w = self.attn(h, h, h, need_weights=need_w, average_attn_weights=False)
        # tokens = tokens + out                                 

        # gene_out = tokens[:, :G, :]                           # (B,G,D)
        # F = gene_out.mean(dim=1)                              # (B,D)

        # if not return_heatmap:
        #     return F
        out, _ = self.attn(h, h, h, need_weights=False)
        tokens = tokens + out                                 

        gene_out = tokens[:, :G, :]                           # (B,G,D)
        F = gene_out.mean(dim=1)                              # (B,D)
        # heat = attn_w[:, :, G:G+self.k_register, :G]          # (B,heads,k,G)
        # heat = heat.mean(dim=1)                               # (B,k,G)
        # print(F.shape)
        return F

# class RegisterSNN(nn.Module):
#     def __init__(self, omic_input_dim: int, model_size_omic="small", k_register: int = 8):
#         super().__init__()
#         self.k_register = k_register
#         self.register = nn.Parameter(torch.randn(1, k_register))
#         dim_dict = {"small": 4096}
#         self.base_dim = dim_dict[model_size_omic]
#         self.in_dim = omic_input_dim + k_register
#         self.hid_dim = self.base_dim + k_register

#         blocks = [SNN_Block(self.in_dim, self.hid_dim)]
#         for _ in range(3):
#             blocks.append(SNN_Block(self.hid_dim, self.hid_dim, dropout=0.25))
#         self.fc_omic = nn.Sequential(*blocks)

#         self.apply(init_non_llm)
#         nn.init.normal_(self.register, 0.0, 0.02)

#     def forward(self, x):
#         # x: (B,G)
#         if x.dim() == 1:
#             x = x.unsqueeze(0)
#         B = x.size(0)
#         r = self.register.expand(B, -1)      # (B,k)
#         x = torch.cat([x, r], dim=1)          # (B,G+k)
#         h = self.fc_omic(x)                   # (B,base+k)
#         return h[:, :-self.k_register]        # (B,base)


class TextEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.enc = TransformerBlock(d_model, n_heads, p_drop=dropout)
        self.apply(init_non_llm)

    def forward(self, x, attn_mask=None):
        # x: (N,T,D)
        h = self.enc(x, attn_mask)
        return h.mean(dim=1)                  # (N,D)


class FrozenLLaMAEmbCPU(nn.Module):
    def __init__(self, llama_path: str):
        super().__init__()
        tok = AutoTokenizer.from_pretrained(llama_path, use_fast=True)
        tok.pad_token = tok.eos_token
        self.tokenizer = tok
        self.llama = AutoModel.from_pretrained(llama_path).cpu().eval()
        for p in self.llama.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def embed_prompts_cpu(self, prompts: list[str]):
        tok = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        out = self.llama(**tok)
        return out.last_hidden_state.cpu()    # (N,T,D)


_LLM_EMB_CACHE = {}


def _hash_key(llama_path: str, prompts: list[str]):
    h = hashlib.sha256()
    h.update(llama_path.encode())
    for p in prompts:
        h.update(b"\n")
        h.update(p.encode())
    return h.hexdigest()


def get_llm_emb_cpu(llama_path: str, prompts: list[str], cache_path: str | None):
    key = _hash_key(llama_path, prompts)
    if key in _LLM_EMB_CACHE:
        return _LLM_EMB_CACHE[key]
    if cache_path is not None and os.path.isfile(cache_path):
        emb = torch.load(cache_path, map_location="cpu")
        _LLM_EMB_CACHE[key] = emb
        return emb
    emb = FrozenLLaMAEmbCPU(llama_path).embed_prompts_cpu(prompts).contiguous()
    _LLM_EMB_CACHE[key] = emb
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(emb, cache_path)
    return emb


class GLA(nn.Module):
    def __init__(
        self,
        gene_dim: int,
        llama_path: str,
        k_register: int = 8,
        V_len: int = 8,
        text_enc_heads: int = 2,
        tau: float = 1.0,
        dropout: float = 0.0,
        n_classes: int = 4,
        llm_cache_path: str | None = None,
    ):
        super().__init__()
        self.tau = tau

        self.cancer_types = {
            "CESC": "Cervical squamous cell carcinoma and endocervical adenocarcinoma",
            "LIHC": "Liver hepatocellular carcinoma",
            "BRCA": "Breast invasive carcinoma",
            "COAD": "Colon adenocarcinoma",
            "KIRC": "Kidney renal clear cell carcinoma",
        }

        self.gene_tok = RegisterSNN(gene_dim, k_register=k_register)

        prompts = self.build_prompts()
        llm_emb_cpu = get_llm_emb_cpu(llama_path, prompts, llm_cache_path)
        self.register_buffer("llm_emb_cpu", llm_emb_cpu, persistent=True)

        D_llm = llm_emb_cpu.size(-1)

        self.gene_to_llm = nn.Linear(self.gene_tok.base_dim, D_llm)
        self.V = nn.Parameter(torch.randn(len(self.cancer_types), V_len, D_llm))
        self.text_enc = TextEncoder(D_llm, text_enc_heads, dropout)
        self.classifier = nn.Linear(D_llm, n_classes)

        self.apply(init_non_llm)
        nn.init.normal_(self.V, 0.0, 0.02)

    def build_prompts(self):
        return ["This is genetic data about " + c for c in self.cancer_types.values()]

    def forward(self, **kwargs):
        x = kwargs["x_omic"]
        # y = kwargs.get("label", None)
        y = torch.tensor([0]).cuda()
        concat = kwargs.get("concat", False)
        return_feat = kwargs.get("return_feat", False)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        device = x.device
        N = self.llm_emb_cpu.size(0)

        F_gene = self.gene_to_llm(self.gene_tok(x))    # (B,D)

        emb_all = self.llm_emb_cpu.to(device)          # (N,T,D)

        texts = []
        for i in range(N):
            Vi = self.V[i].unsqueeze(0)                
            t = torch.cat([emb_all[i:i+1], Vi], dim=1) 
            texts.append(self.text_enc(t))             
        T = torch.cat(texts, dim=0)                    

        F_gene_align = F_gene.detach()
        # logits_align = (F.normalize(F_gene_align, -1) @ F.normalize(T, -1).t()) / self.tau
        logits_align = (F.normalize(F_gene_align, dim=-1) @ F.normalize(T, dim=-1).t()) / self.tau

        # if y is not None:
        #     # print(y)
        #     loss_align = F.cross_entropy(logits_align, y)
        #     print(loss_align.item())
        #     F_out = F_gene + self.V[y].mean(dim=1)     # (B,D)
        # else:
        #     print(y)
        #     F_out = F_gene

        loss_align = F.cross_entropy(logits_align, y)
        F_context = self.V[y].mean(dim=1)
        if concat:
            F_out = torch.cat((F_gene,F_context),dim=1)
        else:
            F_out = F_gene + F_context

        if return_feat:
            return F_out, loss_align

        logits = self.classifier(F_out)                # (B,C)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        # return hazards, S, Y_hat
        return hazards, S, (Y_hat, loss_align)
