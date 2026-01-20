from PFR_files import PFR_model
import jax.numpy as jnp
import os, time
from tqdm import tqdm
import torch

NAME = "CESC"
PCA_Q = 5
K_CLUSTERS = 1500
KNN_K = 5
MAX_ITER = 300
MAX_DATA_RATIO = 0.01
GD_ITERS = 50
LR = 0.01

pfr = PFR_model.PFR(uniform_low=-1, uniform_high=1, uniform_start_size=20, dim=PCA_Q)

folder = f"/vip_media/SharedData/SurvivalPred/data/{NAME}/WSI/pt_files"
save_folder = f"./PFR_result/{NAME}"
os.makedirs(save_folder, exist_ok=True)

def _stat(x: torch.Tensor):
    x = x.detach().float().cpu()
    return f"shape={tuple(x.shape)} nan={bool(torch.isnan(x).any())} min={x.min().item():.3g} max={x.max().item():.3g}"

for fname in tqdm(sorted(os.listdir(folder)), desc=f"PFR ({NAME})"):
    path_pth = os.path.join(folder, fname)
    t0 = time.time()

    emb_train, emb_X, train_pca, X_pca = pfr.read_data_from_pytorch_pth(path_pth, path_pth, q=PCA_Q)
    print(f"\n=== {fname} ===")
    print(f"[Raw]  emb_train: {_stat(emb_train)} | train_pca: {_stat(train_pca)}")

    # obtain centroids C={c_i} and clusters {Q_i}
    # >>> Can be replaced by PGCN/PatchGCN outputs: (C_centroids, clusters_dict) <<<
    C_centroids_X, C_centroids_C, clusters_X, clusters_C = pfr.quantize(train_pca, X_pca, k=K_CLUSTERS)

    X = jnp.array(C_centroids_X.cpu())
    C = jnp.array(C_centroids_C.cpu())

    non_empty = sum(1 for i in range(K_CLUSTERS) if len(clusters_C[i]["points"]) > 0)
    print(f"[Clust] K={K_CLUSTERS} non_empty={non_empty} empty={K_CLUSTERS-non_empty}")

    W_all, kl_trace, _ = pfr.fit(
        C, X,
        max_iter=MAX_ITER, k=KNN_K,
        stop_criterion="data_size", max_data_size=MAX_DATA_RATIO,
        v_init="jump", resets_allowed=False,
        grad_desc_iter=GD_ITERS, lr=LR
    )

    C_high = W_all[pfr.uniform_start_size:]
    if len(kl_trace) > 0:
        print(f"[PFR] add={len(kl_trace)} KL: {float(kl_trace[0]):.6f} -> {float(kl_trace[-1]):.6f} (min={float(min(kl_trace)):.6f})")
    print(f"[C_high] |C_high|={int(C_high.shape[0])}")

    # Eq.(11): X_p = (⋃_{c_i∈C_high} Q_i) ∪ C_low
    Q_high = pfr.my_explode(C_high, clusters_C)
    C_low = [clusters_C[i]["closest_point_idx"] for i in range(K_CLUSTERS) if "closest_point_idx" in clusters_C[i]]
    X_p = sorted(set(Q_high) | set(C_low))

    N = emb_train.shape[0]
    print(f"[Select] |Q_high|={len(Q_high)} |C_low|={len(C_low)} |X_p|={len(X_p)} ratio={len(X_p)/max(N,1):.4f}")

    selected = emb_train[X_p]
    save_path = os.path.join(save_folder, fname)
    torch.save(selected, save_path)

    print(f"[Save] {tuple(selected.shape)} -> {save_path}")
    print(f"[Time] {time.time()-t0:.2f}s")
