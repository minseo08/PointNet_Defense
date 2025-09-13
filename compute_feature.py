import os
import torch
import numpy as np
import argparse
from models import *
from utils import *
from dataset import ModelNet40Dataset, load_data
from torch.nn.functional import softmax
from attacks.utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader

def extract_train_features():
    model.eval()

    all_softmax = []
    train_features = []

    softmax_path = os.path.join(train_feature_save_path, "softmax.npy")
    features_path = os.path.join(train_feature_save_path, "features.npy")

    with torch.no_grad():

        for _, data_pair in enumerate(test_loader):

            data, _ = data_pair
            data = data.float().cuda().transpose(1, 2).contiguous()

            logits, features = model(data, return_global=True)
            s = softmax(logits, dim=1)

            all_softmax += [soft.detach().cpu().numpy() for soft in s]
            train_features += [feat.detach().cpu().numpy() for feat in features]

    
    all_softmax = np.array(all_softmax)
    train_features = np.array(train_features)

    np.save(softmax_path, all_softmax) #[2468, 40]
    np.save(features_path, train_features) #[2468, 1024]

    return np.load(features_path)

def extract_adv_features():
    model.eval()
    adv_features = []

    with torch.no_grad():
        for _, data_pair in enumerate(adv_loader):

            data, _ = data_pair
            data = data.float().cuda().transpose(1, 2).contiguous()

            _, features = model(data, return_global=True)
            adv_features += [feat.detach().cpu().numpy() for feat in features]

    
    return np.array(adv_features)


if __name__ == "__main__":
    print("Start!")
    parser = argparse.ArgumentParser(description='3D Point Clouds')
    parser.add_argument('--data_root', type=str,
                        default='data/ModelNet40')
    
    parser.add_argument('--adv_root', type=str,
                        default='data/adv_samples')

    parser.add_argument('--model', type=str, default='pointnet2',
                        choices=['pointnet', 'pointnet2', 'dgcnn', 'pct'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pct]')
    
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Size of batch)')
    
    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')
    
    parser.add_argument('--attack', type=str, default='taof',
                        choices=['no_attack', 'shift', 'add_chamfer', 'add_hausdorff',
                                 'drop100', 'drop200', 'knn', 'uadvpc', 'tadvpc', 'uaof', 'taof'],
                        help='Attack name')
    
    parser.add_argument('--apply_transform', type=str2bool, default=True,
                        help="whether to apply normalization and other test transforms")
    
    args = parser.parse_args()

    BEST_WEIGHTS = f'checkpoints/{args.model.lower()}/{args.model.lower()}_best.pth'

    set_seed(1)

    if args.model.lower() == 'dgcnn':
        model = DGCNN(args.emb_dims, args.k, output_channels=40)

    elif args.model.lower() == 'pointnet':
        model = PointNetCls(k=40, feature_transform=args.feature_transform)

    elif args.model.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=40)

    elif args.model.lower() == 'pct':
        model = PCT(output_channels=40)

    else:
        print('Model not recognized')
        exit(-1)

    model = model.cuda()
    print('Loading weight {}'.format(BEST_WEIGHTS))
    state_dict = torch.load(BEST_WEIGHTS)

    try:
        model.load_state_dict(state_dict)

    except RuntimeError:
        model.module.load_state_dict(state_dict)

    train_feature_save_path = os.path.join("data/features", args.model.lower())
    if not os.path.exists(train_feature_save_path):
        os.makedirs(train_feature_save_path)

    test_transforms = to_tensor_transform
    if args.apply_transform:
        test_transforms = data_transforms["test"][args.model.lower()]

    train_X, train_y = load_data(args.data_root, set="train")
    train_set = ModelNet40Dataset(train_X, train_y, transform=test_transforms)
    test_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)

    train_features = extract_train_features()


    test_transforms = data_transforms["test"][args.model.lower()]
    adv_path = f"{args.adv_root}/{args.attack.lower()}/{args.model.lower()}"
    adv_samples, gt_labels = load_data(adv_path, set="test")
    adv_set = ModelNet40Dataset(adv_samples, gt_labels, transform=test_transforms)
    adv_loader = DataLoader(adv_set, batch_size=args.batch_size, shuffle=False)

    adv_features = extract_adv_features()

    fea_dists_path = f"data/features/{args.model.lower()}/{args.attack.lower()}_fea_dists.npy"

    if os.path.exists(fea_dists_path):
        print(f"Loading precomputed feature distances from {fea_dists_path}")
        fea_dists = np.load(fea_dists_path)
    else:
        print("Computing feature distances...")
        fea_dists = np.zeros((adv_features.shape[0], train_features.shape[0])) #[2468, 9843]
        fea_dists = euclidean_distance(train_features, adv_features)
        np.save(fea_dists_path, fea_dists)
        print(f"Saved feature distances to {fea_dists_path}")