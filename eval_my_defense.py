import torch
import numpy as np
import argparse
from models import *
from utils import *
from defenses.my_defense import MyDefense
from dataset import ModelNet40Dataset, load_data
from torch.utils.data import DataLoader

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from mpl_toolkits.mplot3d import Axes3D  # noqa

# MODELNET40_CLASSES = [
#     "airplane", "bathtub", "bed", "bench", "bookshelf",
#     "bottle", "bowl", "car", "chair", "cone",
#     "cup", "curtain", "desk", "door", "dresser",
#     "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
#     "laptop", "mantel", "monitor", "night_stand", "person",
#     "piano", "plant", "radio", "range_hood", "sink",
#     "sofa", "stairs", "stool", "table", "tent",
#     "toilet", "tv_stand", "vase", "wardrobe", "xbox"
# ]

# def ensure_preds(prob_or_pred):
#     prob_or_pred = np.asarray(prob_or_pred)
#     if prob_or_pred.ndim == 2:
#         return prob_or_pred.argmax(axis=1)
#     elif prob_or_pred.ndim == 1:
#         return prob_or_pred
#     else:
#         raise ValueError("Shape must be (N,) or (N,C).")

# def class_name(idx, class_names=None):
#     if class_names is None:
#         return str(idx)
#     return class_names[int(idx)]

# def plot_accuracy_bars(gt, pred_no_def, pred_def, save_path):
#     pred_no_def = ensure_preds(pred_no_def)
#     pred_def    = ensure_preds(pred_def)
#     acc_no_def  = (pred_no_def == gt).mean()
#     acc_def     = (pred_def == gt).mean()

#     plt.figure()
#     plt.bar(["No Defense", "My Defense"], [acc_no_def, acc_def])
#     plt.ylabel("Accuracy")
#     plt.title("Accuracy Comparison (Higher is better)")
#     for i, v in enumerate([acc_no_def, acc_def]):
#         plt.text(i, v, f"{v*100:.2f}%", ha="center", va="bottom")
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.close()

# def plot_confusion(gt, pred, save_path, title="Confusion Matrix",
#                    class_names=None, normalize=True,
#                    tick_step=None, max_label_len=12, fontsize=8):
#     """
#     tick_step: 축 라벨을 간격 두고 표시 (예: 2면 2개마다 하나 표시). None이면 자동 결정.
#     max_label_len: 너무 긴 클래스명은 줄임표(...) 처리.
#     fontsize: 축 라벨 글자 크기.
#     """
#     import numpy as np
#     from sklearn.metrics import confusion_matrix
#     import matplotlib.pyplot as plt
#     import itertools

#     pred = ensure_preds(pred)
#     cm = confusion_matrix(gt, pred)
#     if normalize:
#         cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-12)

#     n = cm.shape[0]

#     # --- 라벨 전처리(줄임표/언더스코어 처리) ---
#     if class_names is None:
#         labels = [str(i) for i in range(n)]
#     else:
#         labels = []
#         for s in class_names:
#             s = s.replace("_", " ")
#             if len(s) > max_label_len:
#                 s = s[:max_label_len-1] + "…"
#             labels.append(s)

#     # --- 큰 그림 사이즈 (클래스 수에 비례) ---
#     fig_w = max(8, n * 0.35)
#     fig_h = max(6, n * 0.35)
#     fig = plt.figure(figsize=(fig_w, fig_h))
#     ax = fig.add_subplot(111)

#     im = ax.imshow(cm, interpolation="nearest")
#     ax.set_title(title, fontsize=fontsize+2)
#     ax.set_xlabel("Predicted", fontsize=fontsize+1)
#     ax.set_ylabel("True", fontsize=fontsize+1)

#     # --- 축 라벨 샘플링 (너무 많으면 간격 표시) ---
#     if tick_step is None:
#         # 클래스 많으면 간격 자동 증가
#         tick_step = 1 if n <= 20 else (2 if n <= 40 else 3)

#     ticks = np.arange(n)
#     show_mask = (ticks % tick_step == 0)

#     ax.set_xticks(ticks[show_mask])
#     ax.set_yticks(ticks[show_mask])
#     ax.set_xticklabels([labels[i] for i in ticks[show_mask]], rotation=45, ha="right", fontsize=fontsize)
#     ax.set_yticklabels([labels[i] for i in ticks[show_mask]], fontsize=fontsize)

#     # --- 셀 값 텍스트 (작게/희미하게) ---
#     fmt = ".2f" if normalize else "d"
#     cm_disp = cm if n <= 25 else None  # 너무 크면 생략
#     if cm_disp is not None:
#         for i, j in itertools.product(range(n), range(n)):
#             ax.text(j, i, format(cm[i, j], fmt),
#                     ha="center", va="center", fontsize=fontsize-2)

#     fig.tight_layout()
#     fig.savefig(save_path, bbox_inches="tight")
#     plt.close(fig)


# def show_pc_sample(pc_xyz, save_path, title=""):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     ###
#     # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc_xyz))
#     # o3d.io.write_point_cloud(save_path,pcd)
#     ###

#     X, Y, Z = pc_xyz[:,0], pc_xyz[:,1], pc_xyz[:,2]
#     ax.scatter(X, Y, Z, s=2)
#     ax.set_title(title)

#     max_range = (pc_xyz.max(axis=0) - pc_xyz.min(axis=0)).max()
#     mid = pc_xyz.mean(axis=0)
#     ax.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
#     ax.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
#     ax.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.close()

# def visualize_samples(adv_samples, gt, pred_no_def, pred_def, out_dir, class_names=None, k=8, mode="wrong_only"):
#     os.makedirs(out_dir, exist_ok=True)
#     pred_no_def = ensure_preds(pred_no_def)
#     pred_def    = ensure_preds(pred_def)
#     gt = np.asarray(gt)

#     correct_mask = (pred_def == gt)
#     idx_pool = np.arange(len(gt))
#     if mode == "correct_only":
#         idx_pool = idx_pool[correct_mask]
#     elif mode == "wrong_only":
#         idx_pool = idx_pool[~correct_mask]

#     if len(idx_pool) == 0:
#         print(f"[{mode}] 샘플이 없습니다.")
#         return

#     np.random.seed(0)
#     pick = np.random.choice(idx_pool, size=min(k, len(idx_pool)), replace=False)

#     for i in pick:
#         t = class_name(gt[i], class_names)
#         p0 = class_name(pred_no_def[i], class_names)
#         p1 = class_name(pred_def[i], class_names)
#         title = f"Idx {i} | GT: {t} | NoDef: {p0} | MyDef: {p1}"
#         save_path = os.path.join(out_dir, f"sample_{i}.png")
#         show_pc_sample(adv_samples[i], save_path=save_path, title=title)
#         # save_path = os.path.join(out_dir, f"sample_{i}.pcd")
#         # show_pc_sample(adv_samples[i], save_path=save_path, title=title)


def extract_adv_predictions():
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for _, data_pair in enumerate(adv_loader):
            data, label = data_pair
            data = data.float().cuda().transpose(1, 2).contiguous()

            output = model(data)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            pred = logits.argmax(dim=1)
            all_preds.append(pred.detach().cpu().numpy())
            all_labels.append(label.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_preds, all_labels


if __name__ == "__main__":

    print("Start!")
    parser = argparse.ArgumentParser(description='3D Point Clouds')
    parser.add_argument('--train_path', type=str,
                        default='data/ModelNet40')
    
    parser.add_argument('--adv_root', type=str,
                        default='data/adv_samples')
    
    parser.add_argument('--feature_root', type=str,
                        default='data/features')

    parser.add_argument('--model', type=str, default='pointnet2',
                        choices=['pointnet', 'pointnet2', 'dgcnn', 'pct'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pct]')
    
    parser.add_argument('--attack', type=str, default='drop200',
                        choices=['no_attack', 'shift', 'add_chamfer', 'add_hausdorff',
                                 'drop100', 'drop200', 'knn', 'uadvpc', 'tadvpc', 'uaof', 'taof'],
                        help='Attack name')

    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Size of batch)')
    
    parser.add_argument('--num_nearest', type=int, default=5,
                        help="Num of nearest neighbors to use in KNN-Defense")

    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    
    parser.add_argument('--weight_mode', type=str, default='rank', choices=['uniform', 'inverse', 'softmax', 'gaussian', 'rank'],
                        help='weight mode to use when computing weight from distance')

    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    
    parser.add_argument('--tau', type=float, default=0.1, metavar='N',
                        help='tau for softmax weight mode')
                        
    args = parser.parse_args()
    BEST_WEIGHTS = f'checkpoints/{args.model.lower()}/{args.model.lower()}_best.pth'
    #BEST_WEIGHTS = f''

    set_seed(1)
    print(args)


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

    test_transforms = data_transforms["test"][args.model.lower()]

    adv_path = f"{args.adv_root}/{args.attack.lower()}/{args.model.lower()}"
    adv_samples, gt_labels = load_data(adv_path, set="test")
    adv_set = ModelNet40Dataset(adv_samples, gt_labels, transform=test_transforms)
    adv_loader = DataLoader(adv_set, batch_size=args.batch_size, shuffle=False)

    softmax_path = f"data/features/{args.model.lower()}/softmax.npy"
    geo_dist_path = f"data/features/{args.model.lower()}/{args.attack.lower()}_geo_dists.npy"
    fea_dist_path = f"data/features/{args.model.lower()}/{args.attack.lower()}_fea_dists.npy"

    train_softmax = np.load(softmax_path)
    geo_dists = np.load(geo_dist_path) # [2468, 9843]
    fea_dists = np.load(fea_dist_path)

    fea_dists_norm = (fea_dists - fea_dists.min()) / (fea_dists.max() - fea_dists.min())
    geo_dists_norm = (geo_dists - geo_dists.min()) / (geo_dists.max() - geo_dists.min())

    my_defense = MyDefense()

    P_fea = my_defense.knn_proba_from_dists(fea_dists_norm, train_softmax, k=args.num_nearest, weight_mode=args.weight_mode, tau=args.tau)  # [N, C]
    P_geo  = my_defense.knn_proba_from_dists(geo_dists_norm, train_softmax, k=args.num_nearest, weight_mode=args.weight_mode, tau=args.tau)  # [N, C]

    #P_fused, alpha = my_defense.fuse(P_fea, P_geo)   # alpha_vec: [N], 각 샘플의 가중치
    P_fused_mean, alpha_1 = my_defense.fuse_mean(P_fea, P_geo)
    preds_mean = P_fused_mean.argmax(axis=1)
    acc1 = (preds_mean == gt_labels).sum() / len(gt_labels)
    print("alpha: ", alpha_1)
    print(acc1)

    P_fused_median, alpha_2 = my_defense.fuse_median(P_fea, P_geo)
    preds_median = P_fused_median.argmax(axis=1)
    acc2 = (preds_median == gt_labels).sum() / len(gt_labels)
    print("alpha: ", alpha_2)
    print(acc2)

    P_fused, alpha_3 = my_defense.fuse_logop(P_fea, P_geo)
    preds = P_fused.argmax(axis=1)
    acc3 = (preds == gt_labels).sum() / len(gt_labels)
    print("alpha: ", alpha_3)
    print(acc3)

    acc = max(acc1, acc2, acc3)
    print("My Defense Accuracy: {:.4f}".format(100 * acc))


    adv_preds, adv_true = extract_adv_predictions()
    acc_no_def = (adv_preds == adv_true).sum() / len(adv_true)
    print("No Defense Accuracy: {:.4f}".format(100 * acc_no_def))

    # save_dir = "vis_results_uaof"
    # os.makedirs(save_dir, exist_ok=True)

    #plot_accuracy_bars(gt_labels, adv_preds, preds_mean, save_path=os.path.join(save_dir, "accuracy_bar.png"))
    # plot_confusion(gt_labels, adv_preds, class_names=MODELNET40_CLASSES, save_path=os.path.join(save_dir, "cm_no_def.png"), title="No Defense CM")
    # plot_confusion(gt_labels, preds_mean, class_names=MODELNET40_CLASSES, save_path=os.path.join(save_dir, "cm_my_def.png"), title="My Defense CM")
    #visualize_samples(adv_samples, gt_labels, adv_preds, preds_mean, class_names=MODELNET40_CLASSES, out_dir=os.path.join(save_dir, "samples"), k=50)

    # H_feat = entropy(P_feat); H_geo = entropy(P_geo)
    # print("Mean entropy - feat: {:.3f}, geo: {:.3f}".format(H_feat.mean(), H_geo.mean()))
    # print("alpha range: [{:.3f}, {:.3f}] mean: {:.3f}".format(alpha_vec.min(), alpha_vec.max(), alpha_vec.mean()))