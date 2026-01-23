import jax.numpy as jnp
from jax import grad
import random
import jax
import torch
import numpy as np
from .PFR_super import PFR_super
import numpy
import torch
from torch.nn import functional as F
from tqdm import tqdm 

class PFR(PFR_super):
    def __init__(self, uniform_low=-1, uniform_high=1, uniform_start_size=20, dim=768):
        super().__init__()
        self.uniform_low = uniform_low
        self.uniform_high = uniform_high
        self.uniform_start_size = uniform_start_size
        self.dim = dim
        self.random_init = False

    def _get_nearest(self, sample, point):
        norms = jnp.linalg.norm(sample - point, axis=1)
        return jnp.argsort(norms)[0]

    def _knn(self, x, y, k, last_only, discard_nearest, avg):
        dist_x = jnp.sum((x ** 2), axis=-1)[:, jnp.newaxis]
        dist_y = jnp.sum((y ** 2), axis=-1)[:, jnp.newaxis].T
        cross = - 2 * jnp.matmul(x, y.T)
        distmat = dist_x + cross + dist_y
        distmat = jnp.clip(distmat, 1e-10, 1e+20)

        if discard_nearest:
            if not avg:
                knn, _ = jax.lax.top_k(-distmat, k + 1)
            else:
                knn = -jnp.sort(distmat)
            knn = knn[:, 1:]
        else:
            knn = -distmat

        if last_only:
            knn = knn[:, -1:]

        return jnp.sqrt(-knn)

    def _kl_divergence_knn(self, x, y, k, eps, discard_nearest_for_xy):
        n, d = x.shape
        m, _ = y.shape
        nns_xx = self._knn(x, x, k=k, last_only=True, discard_nearest=True, avg=False)
        nns_xy = self._knn(x, y, k=m, last_only=False, discard_nearest=discard_nearest_for_xy, avg=discard_nearest_for_xy)
        p = nns_xx
        q = nns_xy
        log_r = p - q
        r = jnp.exp(log_r)
        k = r * log_r - (r - 1)
        kl_estimate = jnp.mean(k)
        divergence = kl_estimate
        return divergence

    def calculate_statistical_distance(self, x, y, k=5, eps=1e-8, discard_nearest_for_xy=False):
        return self._kl_divergence_knn(x, y, k, eps, discard_nearest_for_xy)

    def gradient_descend(self, X, W, v, scaling_factor, max_iterations, lr=0.01, k=5, discard_nearest_for_xy=False):
        i = 0
        while i < max_iterations:
            gradient = grad(lambda v: self.calculate_statistical_distance(X, jnp.concatenate((W, v[jnp.newaxis, :])), k, discard_nearest_for_xy=discard_nearest_for_xy))(v)
            v = v - lr * scaling_factor * gradient
            i += 1
        return v

    def _get_uniform_start(self, do_normalize):
        def normalize(v):
            norm = np.linalg.norm(v)
            if norm == 0:
                return v
            return v / norm
        if do_normalize:
            return jnp.array([normalize(each) for each in np.random.uniform(low=self.uniform_low,high=self.uniform_high,size=(self.uniform_start_size,self.dim))])
        else:
            return jnp.array([each for each in np.random.uniform(low=self.uniform_low,high=self.uniform_high,size=(self.uniform_start_size,self.dim))])

    def fit(self, train, X, D=None, k=5, max_iter=100, stop_criterion="increase", min_difference=0, resets_allowed=False, max_resets=2, max_data_size=1, min_kl=0, max_sequential_increases=3, random_init_pct=0, random_restart_prob=0, scale_factor="auto", v_init='mean', grad_desc_iter=50, discard_nearest_for_xy=False, normalize=True, lr=0.01):
        if not random_init_pct and D is None:
            W = self._get_uniform_start(normalize)
            self.random_init = True
        elif D is None:
            amount = int(random_init_pct * len(train))
            W = jnp.array(random.sample(train.tolist(), amount))
        else:
            W = D[:]

        kl_dist_prev = self.calculate_statistical_distance(X, W, k, discard_nearest_for_xy=discard_nearest_for_xy)

        print("Starting KL: " + str(kl_dist_prev))
        if v_init == 'mean' or v_init == 'prev_opt':
            v = jnp.mean(X, axis=0)
        elif v_init == 'jump':
            v = jnp.array(random.sample(X.tolist(), 1)).squeeze()
        adder = train[:]
        kl_divs = []

        scale_factor = jnp.linalg.norm(v)/jnp.linalg.norm(grad(lambda v: self.calculate_statistical_distance(X, jnp.concatenate((W, v[jnp.newaxis, :])), k, discard_nearest_for_xy=discard_nearest_for_xy))(v)) if scale_factor == "auto" else scale_factor

        i = 0
        just_reset = False
        num_resets = 0
        total_iter = 0
        increases = 0
        while True:
            if i == 0 or just_reset or random.random() < random_restart_prob:
                v = self.gradient_descend(X, W, v, scale_factor, grad_desc_iter * 3, lr=lr, k=k, discard_nearest_for_xy=discard_nearest_for_xy)
            else:
                v = self.gradient_descend(X, W, v, scale_factor, grad_desc_iter, lr=lr, k=k, discard_nearest_for_xy=discard_nearest_for_xy)
            idx = self._get_nearest(v, adder)
            minvals = adder[idx]
            adder = jnp.delete(adder, idx, axis=0)

            W_tmp = jnp.concatenate((W, jnp.array(minvals)[jnp.newaxis, :]))

            kl_dist = self.calculate_statistical_distance(X, W_tmp, k, discard_nearest_for_xy=discard_nearest_for_xy)
            print("PFR-KL at iter " + str(i) + ": " + str(kl_dist))

            if total_iter > max_iter:
                break

            if v_init == 'mean':
                v = jnp.mean(X, axis=0)
            elif v_init == 'jump':
                v = jnp.array(random.sample(X.tolist(), 1)).squeeze()

            adder, i, just_reset, stop, v, increases, num_resets = self._test_stop_criterion(v_init, stop_criterion, kl_dist, kl_dist_prev, num_resets, max_resets, min_difference, increases, max_sequential_increases, min_kl, max_data_size, train, X, i, v, just_reset, resets_allowed, adder)

            if stop:
                break
            if not just_reset:
                W = W_tmp
                kl_divs += [kl_dist]
                kl_dist_prev = kl_dist
                i += 1
                total_iter += 1
        return W, kl_divs, (v, scale_factor, just_reset, num_resets, increases, adder, kl_divs)

    def _test_stop_criterion(self, v_init, stop_criterion, kl_dist, kl_dist_prev, num_resets, max_resets, min_difference, increases, max_sequential_increases, min_kl, max_data_size, train, X, i, v, just_reset, resets_allowed, adder):
        stop = False
        if stop_criterion == "increase" and kl_dist - kl_dist_prev > 0:
            stop = True
        elif stop_criterion == "max_resets" and kl_dist - kl_dist_prev > 0 and num_resets == max_resets:
            stop = True
        elif stop_criterion == "min_difference" and kl_dist_prev - kl_dist < min_difference:
            stop = True
        elif stop_criterion == 'sequential_increase_tolerance' and kl_dist - kl_dist_prev > 0 and increases == max_sequential_increases:
            stop = True
        elif stop_criterion == 'min_kl' and kl_dist < min_kl:
            stop = True
        elif stop_criterion == 'data_size' and i > int(max_data_size * len(train)):
            stop = True
        if stop:
            if just_reset:
                increases += 1
            if resets_allowed and num_resets < max_resets:
                num_resets += 1
                if v_init == 'prev_opt':
                    v = jnp.mean(X, axis=0)
                print("KL Div Increase, Resetting G")
                adder = train[:]
                i -= 1
                stop = False
            just_reset = True
        else:
            just_reset = False
            increases = 0
        return adder, i, just_reset, stop, v, increases, num_resets

    def _return_kmeans(self, df, k, rseed):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=rseed).fit(df)
        return kmeans

    def kmeans_pytorch(self, X, k, random_state=None):
        N, D = X.shape
        if k >= N:
            clusters_dict = {
                i: {
                    'centroid': X[i],
                    'points': [i],
                    'closest_point_idx': i
                }
                for i in range(N)
            }
            return X.clone(), clusters_dict

        if random_state is not None:
            torch.manual_seed(random_state)
        X_mean = X.mean(0, keepdim=True)
        X_std = X.std(0, keepdim=True)
        X_normalized = (X - X_mean) / X_std
        center_index = torch.randperm(N)[:k] 
        centroids = X[center_index].clone()
        
        for _ in tqdm(range(20), desc="Preprocessing"):
            distances = torch.sum((X[:, None, :] - centroids[None, :, :]) ** 2, dim=-1)
            clusters = torch.argmin(distances, dim=1)
            new_centroids = torch.stack([X[clusters == i].mean(0) for i in range(k)])
            if torch.max(torch.abs(new_centroids - centroids)) < 1e-6:
                break
            centroids = new_centroids
            
        clusters_dict = {i: {'centroid': None, 'points': []} for i in range(k)}
        for idx, cluster in enumerate(clusters):
            clusters_dict[cluster.item()]['points'].append(idx)

        for i in range(k):
            clusters_dict[i]['centroid'] = centroids[i]
            point_indices = clusters_dict[i]['points']
            if len(point_indices) > 0:
                cluster_points = X[point_indices]
                distances_to_centroid = torch.sum((cluster_points - centroids[i]) ** 2, dim=1)
                min_distance_idx = torch.argmin(distances_to_centroid).item()
                clusters_dict[i]['closest_point_idx'] = point_indices[min_distance_idx]
        return centroids, clusters_dict

    def quantize(self, df_train, df_x, k=1500, rseed='auto', rseed1=234, rseed2=456):
        from sklearn.cluster import KMeans
        if rseed == 'auto':
            rseed1 = random.randint(0,100000)
            rseed2 = random.randint(0,100000)

        # center_X, dict_X -> can also read from PGCN
        # center_train, dict_train -> can also read from PGCN
        center_X, dict_X = self.kmeans_pytorch(df_train,k,rseed1)
        center_train, dict_train = self.kmeans_pytorch(df_x,k,rseed2)
        return center_X, center_train, dict_X, dict_train


    def read_data_from_pt(self, path):
        data = torch.load(path)
        return data.cuda()

    def read_data_from_pytorch_pth(self, path_train, path_X, q=8):
        train_with_embeddings = self.read_data_from_pt(path_train)
        X_with_embeddings = self.read_data_from_pt(path_X)
        U, S, Vh = torch.pca_lowrank(train_with_embeddings, q=q)
        train_with_embeddings_output_tensor = torch.matmul(train_with_embeddings, Vh)
        U, S, Vh = torch.pca_lowrank(X_with_embeddings, q=q)
        X_with_embeddings_output_tensor = torch.matmul(X_with_embeddings, Vh)
        return train_with_embeddings, X_with_embeddings, train_with_embeddings_output_tensor, X_with_embeddings_output_tensor

    def my_explode(self,chosen_centroids,dict_train):
        all_centroids = jnp.stack([jnp.array(item['centroid'].cpu()) for item in dict_train.values()])
        paired = []
        for chosen_center in chosen_centroids:
            matching_indices = jnp.where(jnp.all(all_centroids == chosen_center, axis=1))[0]
            for idx in matching_indices:
                idx = idx.item()
                paired.extend(dict_train[idx]['points'])
        return paired

