import numpy as np

# 파일 불러오기
labels = np.load("data/adv_samples/drop100/pointnet2/test/labels.npy", allow_pickle=True)
pointclouds = np.load("data/adv_samples/drop100/pointnet2/test/pointclouds.npy", allow_pickle=True)
# targets = np.load("data/adv_samples/drop100/pointnet/test/targets.npy", allow_pickle=True)

# 하나의 .npz 파일로 저장
np.savez("../data/drop1002.npz", test_pc=pointclouds, test_label=labels)

print("저장 완료!")
