import os
import torch
import numpy as np
import argparse
from models import *
from utils import *
from dataset import ModelNet40Dataset, load_data
from attacks.utils import *
from tqdm import tqdm


def compute_geometric_dists(adv_pcs_np, train_pcs_np, batch_size=64, device="cuda"):
    """
    Compute Chamfer distances between adv_pcs and train_pcs using ChamferDist (Torch).
    Args:
        adv_pcs_np: [N_test, P, 3] (numpy)
        train_pcs_np: [N_train, P, 3] (numpy)
    Returns:
        dists: [N_test, N_train] (numpy)
    """
    chamfer_metric = ChamferDist(method='both').to(device)
    N_test, P, _ = adv_pcs_np.shape
    N_train = train_pcs_np.shape[0]

    adv_pcs = torch.from_numpy(adv_pcs_np).float().to(device)
    train_pcs = torch.from_numpy(train_pcs_np).float().to(device)

    dists = torch.zeros((N_test, N_train), dtype=torch.float32)

    print("Computing Chamfer distance matrix...")
    for i in tqdm(range(N_test), desc="Test set"):
        pc1 = adv_pcs[i].unsqueeze(0).repeat(batch_size, 1, 1)  # [B, P, 3]

        for j in range(0, N_train, batch_size):
            end = min(j + batch_size, N_train)
            pc2 = train_pcs[j:end]  # [B2, P, 3]
            B2 = pc2.size(0)
            pc1_batch = pc1[:B2]

            loss = chamfer_metric(pc1_batch, pc2, batch_avg=False)  # [B]
            dists[i, j:j+B2] = loss.detach().cpu()

    return dists.numpy()


if __name__ == "__main__":
    print("Start!")
    parser = argparse.ArgumentParser(description='3D Point Clouds')
    
    parser.add_argument('--adv_root', type=str,
                        default='data/adv_samples')

    parser.add_argument('--model', type=str, default='pointnet2',
                        choices=['pointnet', 'pointnet2', 'dgcnn', 'pct'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pct]')
    
    parser.add_argument('--attack', type=str, default='taof',
                        choices=['no_attack', 'shift', 'add_chamfer', 'add_hausdorff',
                                 'drop100', 'drop200', 'knn', 'uadvpc', 'tadvpc', 'uaof', 'taof'],
                        help='Attack name')
                        
    args = parser.parse_args()

    test_transforms = data_transforms["test"][args.model.lower()]
    adv_path = f"{args.adv_root}/{args.attack.lower()}/{args.model.lower()}"

    adv_samples, gt_labels = load_data(adv_path, set="test")

    adv_pointclouds = adv_samples

    print("shape of adv point clouds: ", adv_pointclouds.shape)
    train_pointclouds = np.load("data/ModelNet40/train/pointclouds.npy") # [9843, 2048, 3]
    print("shape of train point clouds: ", train_pointclouds.shape)

    geo_dists_path = f"data/features/{args.model.lower()}/{args.attack.lower()}_geo_dists.npy"

    if os.path.exists(geo_dists_path):
        print(f"Loading precomputed geometric distances from {geo_dists_path}")
        geo_dists = np.load(geo_dists_path)
    else:
        print("Computing geometric distances...")
        geo_dists = compute_geometric_dists(
            adv_pointclouds, train_pointclouds, batch_size=64, device="cuda"
        ) # [2468, 9843]
        np.save(geo_dists_path, geo_dists)
        print(f"Saved geometric distances to {geo_dists_path}")