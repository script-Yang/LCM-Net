from PFR_files import PFR_model
import jax.numpy as jnp
import os, time, argparse
from tqdm import tqdm
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
args = parser.parse_args()

NAME = args.name
PCA_Q = 5
K_CLUSTERS = 1500
KNN_K = 5
MAX_ITER = 300
MAX_DATA_RATIO = 0.01
GD_ITERS = 50
LR = 0.01

folder = f"/vip_media/sicheng/DataShare/tmi_re/UNI_results/UNI_{NAME}/pt_files"
save_folder = f"/vip_media/sicheng/DataShare/tmi_re/LCM-Net/PFR_pt_uni/{NAME}"
os.makedirs(save_folder, exist_ok=True)

pfr = PFR_model.PFR(uniform_low=-1, uniform_high=1, uniform_start_size=20, dim=PCA_Q)

def _stat(x: torch.Tensor):
    x = x.detach().float().cpu()
    return f"shape={tuple(x.shape)} nan={bool(torch.isnan(x).any())} min={x.min().item():.3g} max={x.max().item():.3g}"

def _list_src_files(src_dir):
    return sorted([f for f in os.listdir(src_dir) if not f.startswith(".")])

def _cleanup_tmp(dst_dir):
    for f in os.listdir(dst_dir):
        if f.endswith(".tmp"):
            try:
                os.remove(os.path.join(dst_dir, f))
            except:
                pass

_cleanup_tmp(save_folder)

src_files = _list_src_files(folder)
done_files = set([f for f in os.listdir(save_folder) if not f.endswith(".tmp")])
todo_files = [f for f in src_files if f not in done_files]

print(f"[IO] src={folder} total={len(src_files)} done={len(done_files)} todo={len(todo_files)}")
print(f"[IO] dst={save_folder}")

for fname in tqdm(todo_files, desc=f"PFR ({NAME})"):
    path_pth = os.path.join(folder, fname)
    save_path = os.path.join(save_folder, fname)
    tmp_path = save_path + ".tmp"
    t0 = time.time()

    if os.path.exists(save_path):
        continue

    try:
        emb_train, emb_X, train_pca, X_pca = pfr.read_data_from_pytorch_pth(
            path_pth, path_pth, q=PCA_Q
        )

        print(f"\n=== {fname} ===")
        print(f"[Raw] emb_train: {_stat(emb_train)} | train_pca: {_stat(train_pca)}")

        C_centroids_X, C_centroids_C, clusters_X, clusters_C = pfr.quantize(
            train_pca, X_pca, k=K_CLUSTERS
        )

        X = jnp.array(C_centroids_X.detach().cpu().float().numpy())
        C = jnp.array(C_centroids_C.detach().cpu().float().numpy())

        K_eff = min(K_CLUSTERS, len(clusters_C))
        non_empty = sum(1 for i in range(K_eff) if len(clusters_C[i]["points"]) > 0)
        print(f"[Clust] K_req={K_CLUSTERS} K_eff={K_eff} non_empty={non_empty} empty={K_eff-non_empty}")

        W_all, kl_trace, _ = pfr.fit(
            C, X,
            max_iter=MAX_ITER,
            k=KNN_K,
            stop_criterion="data_size",
            max_data_size=MAX_DATA_RATIO,
            v_init="jump",
            resets_allowed=False,
            grad_desc_iter=GD_ITERS,
            lr=LR
        )

        C_high = W_all[pfr.uniform_start_size:]
        if len(kl_trace) > 0:
            print(f"[PFR] add={len(kl_trace)} KL: {float(kl_trace[0]):.6f} -> {float(kl_trace[-1]):.6f}")

        Q_high = pfr.my_explode(C_high, clusters_C)
        C_low = [
            clusters_C[i]["closest_point_idx"]
            for i in range(K_eff)
            if "closest_point_idx" in clusters_C[i]
        ]
        X_p = sorted(set(Q_high) | set(C_low))

        selected = emb_train[X_p]
        torch.save(selected, tmp_path)
        os.replace(tmp_path, save_path)

        print(f"[Save] {tuple(selected.shape)} -> {save_path}")
        print(f"[Time] {time.time()-t0:.2f}s")

    except Exception as e:
        print(f"[Error] {fname}: {repr(e)}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            pass
        continue
