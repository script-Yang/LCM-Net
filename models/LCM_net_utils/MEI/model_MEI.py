import torch
import torch.nn as nn
import torch.nn.functional as F
from models.LCM_net_utils.GLA.model_GLA import GLA

class MLPExpert(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, p_drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(dim * hidden_mult, dim),
        )

    def forward(self, x):  # (B,C)
        return self.net(x)  # (B,C)

class AttnPoolExpert(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, p_drop: float = 0.0):
        super().__init__()
        self.q = nn.Parameter(torch.randn(dim))  # (C,) 
        self.mlp = MLPExpert(dim, hidden_mult=hidden_mult, p_drop=p_drop)

    def forward(self, x_tokens):  # (B,N,C)
        # attn logits: (B,N)
        logits = (x_tokens * self.q.view(1, 1, -1)).sum(dim=-1)
        w = F.softmax(logits, dim=1)  # (B,N)
        pooled = (w.unsqueeze(-1) * x_tokens).sum(dim=1)  # (B,C)
        return self.mlp(pooled)  # (B,C)

class MEI(nn.Module):
    def __init__(self, dim: int, Ng: int = 4, Np: int = 4, Nf: int = 2, p_drop: float = 0.0, n_classes=4,
        gene_in_dim = 4096, path_in_dim = 1024,
    ):
        super().__init__()
        self.Ng, self.Np, self.Nf = Ng, Np, Nf

        self.g_exps = nn.ModuleList([MLPExpert(dim, p_drop=p_drop) for _ in range(Ng)])
        self.p_exps = nn.ModuleList([AttnPoolExpert(dim, p_drop=p_drop) for _ in range(Np)])
        self.f_exps = nn.ModuleList([MLPExpert(dim, p_drop=p_drop) for _ in range(Nf)])

        self.g_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, Ng))
        self.p_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, Np))

        self.f_gate = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, Nf))
        self.f_in = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim))

        self.GLA = GLA(
            gene_dim=60660,                   
            llama_path="/vip_media/sicheng/DataShare/Llama-2-7b-hf",
        )

        self.classifier = nn.Linear(dim, n_classes)

        self.gene_proj = nn.Sequential(
            nn.LayerNorm(gene_in_dim),
            nn.Linear(gene_in_dim, dim),
        )
        self.path_proj = nn.Sequential(
            nn.LayerNorm(path_in_dim),
            nn.Linear(path_in_dim, dim),
        )

    def forward_feat(self, F_gene, F_path):
        if F_gene.dim() == 2:
            # Xg = F_gene  # (B,C)
            Xg = self.gene_proj(F_gene)
        else:
            raise ValueError(f"F_gene must be (B,C), got {tuple(F_gene.shape)}")

        if F_path.dim() == 2:
            Xp = F_path.unsqueeze(0)  # (1,N,C)
        elif F_path.dim() == 3:
            Xp = F_path  # (B,N,C)
        else:
            raise ValueError(f"F_path must be (N,C) or (B,N,C), got {tuple(F_path.shape)}")

        Xp = self.path_proj(Xp)
        B, C = Xg.shape
        assert Xp.shape[0] == B and Xp.shape[2] == C, "Batch/dim mismatch between gene/path"

        wg = F.softmax(self.g_gate(Xg), dim=-1)

        g_stack = torch.stack([exp(Xg) for exp in self.g_exps], dim=1) 
        FGIMF = (wg.unsqueeze(-1) * g_stack).sum(dim=1)  

        Xp_mean = Xp.mean(dim=1) 
        wp = F.softmax(self.p_gate(Xp_mean), dim=-1) 

        p_stack = torch.stack([exp(Xp) for exp in self.p_exps], dim=1) 
        FPIMF = (wp.unsqueeze(-1) * p_stack).sum(dim=1) 

        fused = torch.cat([FGIMF, FPIMF], dim=-1) 
        wf = F.softmax(self.f_gate(fused), dim=-1)  

        fused_in = self.f_in(fused) 
        f_stack = torch.stack([exp(fused_in) for exp in self.f_exps], dim=1)  
        FIMMF = (wf.unsqueeze(-1) * f_stack).sum(dim=1)  

        weights = {"wg": wg, "wp": wp, "wf": wf}
        return FIMMF, weights

    def forward(self, **kwargs):
        x_omic = kwargs["x_omic"]
        x_path = kwargs["x_path"]

        f_omic, loss_align = self.GLA(x_omic=x_omic, return_feat = True)
        f_path = x_path

        FIMMF, weights = self.forward_feat(f_omic, f_path)

        logits = self.classifier(FIMMF) # (B,C)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        # return hazards, S, Y_hat
        return hazards, S, (Y_hat, loss_align)



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 1
    C = 256
    N = 128           
    Cp = 1024         
    F_gene = torch.randn(B, C, device=device)
    F_path_raw = torch.randn(N, Cp, device=device)
    path_proj = nn.Linear(Cp, C).to(device)
    F_path = path_proj(F_path_raw)   # (N, 256)
    model = MEI(
        dim=C,
        Ng=4,     # gene experts
        Np=4,     # pathology experts
        Nf=2,     # fusion experts
        p_drop=0.1
    ).to(device)

    model.eval()
    with torch.no_grad():
        F_out, weights = model(F_gene, F_path)
    print("Output feature shape:", F_out.shape)     # (1, 256)

    print("\nGate weights:")
    print("wg (gene):", weights["wg"].shape, weights["wg"])
    print("wp (path):", weights["wp"].shape, weights["wp"])
    print("wf (fusion):", weights["wf"].shape, weights["wf"])

if __name__ == "__main__":
    main()
