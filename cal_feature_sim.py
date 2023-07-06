import numpy as np
import glob
from scipy.spatial.distance import cdist
import torch

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def chamfer(query, target_feature, comparator=False):
    query = torch.Tensor(query).cuda()
    target_feature = torch.Tensor(target_feature).cuda()
    simmatrix = torch.einsum('ik,jk->ij', [query, target_feature])
    if comparator:
        simmatrix = comparator(simmatrix).detach()
    sim = simmatrix.max(dim=1)[0].sum().cpu().item() / simmatrix.shape[0]
    return sim

# 一、单个测试
# query_file = '/home/chenqingzhong/data/chenqingzhong/Data/zhongshan_endoscope/ENDO_core/features/imac/frames/hfq_1_1_0310_0318.npy'
# # target_file = '/home/chenqingzhong/data/chenqingzhong/Data/zhongshan_endoscope/ENDO_core/features/imac_pca1024/frames/xxy_2_1_0359_0361.npy'
# target_database_folder = '/home/chenqingzhong/data/chenqingzhong/Data/zhongshan_endoscope/ENDO_core/features/imac/frames/'
# video_database_folder = '/home/chenqingzhong/data/chenqingzhong/Data/zhongshan_endoscope/ENDO_background/features/imac/frames/'
# query = np.load(query_file)
# # target = np.load(target_file)
#
# # sim = chamfer(query, target)
# # print(round(sim,3))
#
# video_database = sorted(glob.glob(target_database_folder + 'hfq_2' + '*.*'))
# for video in video_database:
#     # print(video)
#     video = np.load(video)
#     sim = chamfer(query, video)
#     print(round(sim,3))
#
# print('-----')
#
# video_database = sorted(glob.glob(video_database_folder + 'hfq_2' + '*.*'))
# for video in video_database:
#     # print(video)
#     video = np.load(video)
#     # dist = np.nan_to_num(cdist(query, video, metric='euclidean'))
#     # print(-dist.mean())
#     sim = chamfer(query, video)
#     print(round(sim,3))

# 二、批量测试
from numpy import *
features_folder = '/home/chenqingzhong/data/chenqingzhong/code/CoCLR/log-pretrain/infonce_k65536_endo240-128_s3d_bs16_lr0.001_seq2_len32_ds1_e700_schedule450/model/feature_epoch665/' # SOTA=0.315
query_database_folder = '/home/chenqingzhong/data/chenqingzhong/Data/zhongshan_endoscope/polyp_match/query/'
query_database = sorted(glob.glob(query_database_folder + '*.*'))

ap_res = []
rank_res = []

def embedding_distance(feature_1, feature_2):
    dist = np.linalg.norm(feature_1 - feature_2)
    return dist

for query_video in query_database:
    query_id = query_video.split('.')[0].split('/')[-1]
    query = np.load(features_folder + query_id + '.npy')

    patient_id = query_id.split('_')[0]
    firstorsecond = query_id.split('_')[1]
    update_firstorsecond = '2' if firstorsecond == '1' else '1'
    poly_id = query_id.split('_')[2]
    videodb_id_prefix = patient_id + '_' + update_firstorsecond + '_' + poly_id
    query_database = sorted(glob.glob(features_folder + videodb_id_prefix + '*'))

    target_id = sorted(glob.glob(query_database_folder + videodb_id_prefix + '*'))[0].split('.')[0].split('/')[-1]
    target = np.load(features_folder + target_id + '.npy')
    target_sim = round(chamfer(query, target), 6) # 倒角距离
    # target_sim = -(np.nan_to_num(cdist(query, target, metric='euclidean'))[0][0]) # 欧氏距离v1
    # target_sim = -embedding_distance(query, target) # 欧氏距离v2
    # target_sim = 1 - np.nan_to_num(cdist(query, target, metric='cosine'))
    print(query_id + " -> " + target_id)

    # 记录：欧氏距离v1代码可能有问题，baseline上排名糟糕；使用了v2后ch的排名比v1提升明显，甚至比倒角距离也要好，但不能适用于时长不一致的视频片段又是一个新的问题。

    db_res = []
    for video in query_database:
        video = np.load(video)
        sim = round(chamfer(query, video), 6) # 倒角距离
        # sim = -(np.nan_to_num(cdist(query, video, metric='euclidean'))[0][0]) # 欧氏距离v1
        # sim = -embedding_distance(query, video)  # 欧氏距离v2
        # sim = 1 - np.nan_to_num(cdist(query, target, metric='cosine'))
        db_res.append(sim)

    db_res.sort(reverse = -1)
    for i, sim in enumerate(db_res):
        if sim == target_sim:
            rank = i + 1
            print(rank)
            rank_res.append(rank)
            ap_res.append(round(1/rank ,3))
            break

print("AP: " + str(ap_res))
print("rank: " + str(rank_res))
print("mAP: " + str(round(mean(ap_res), 3)))
print("mean rank: " + str(round(mean(rank_res), 3)))
rank_res.sort()
print("sorted rank: " + str(rank_res))

# import torch.nn.functional as F
# import torch
# import os
#
# dirname = '/home/chenqingzhong/data/chenqingzhong/code/CoCLR/log-pretrain/infonce_k65536_endo240-128_s3d_bs16_lr0.001_seq2_len32_ds1_lessaug/model/feature_epoch263/'
#
# device = torch.device('cuda')
# train_feature = torch.load(
#     os.path.join(dirname, 'endo240_train_feature.pth.tar')).to(device)
#
# # centering
# train_feature = train_feature - train_feature.mean(dim=0, keepdim=True)
#
# # normalize
# train_feature = F.normalize(train_feature, p=2, dim=1)
#
# # dot product
# sim = train_feature.matmul(train_feature.t())
#
# # k = 5
# # topkval, topkidx = torch.topk(sim, k, dim=1)
#
# # ch1-1 ch2-1
# # for i in range(49, 93):
# #     print(i, sim[46][i].item())
# # topkval, topkidx = torch.topk(sim[46][49:93], k=5)
#
# # ch2-1 ch1-1
# # for i in range(0, 49):
# #     print(i, sim[85][i].item())
# # topkval, topkidx = torch.topk(sim[85][0:49], k=5)
#
# # cxy_1_1_0528_0535 108(94:94+123)
# # cxy_2_1_0334_0336 250(217:217+143)
# # for i in range(216, 216+143):
# #     print(i, sim[107][i].item())
# # topkval, topkidx = torch.topk(sim[107][216:216+143], k=5)
#
# # ykf_2_1_0227_0228 4859(4697:4697+171)
# # ykf_1_1_0508_0509 4598(4508:4508+189)
# # ranking: 1
# # sim: 0.8322
# # for i in range(4507, 4507+189):
# #     print(i, sim[4858][i].item())
# # topkval, topkidx = torch.topk(sim[4858][4507:4507+189], k=20)
#
# # reverse:
# # ranking: 1
# # sim: 0.8322
# for i in range(4696, 4696+171):
#     print(i, sim[4597][i].item())
# topkval, topkidx = torch.topk(sim[4597][4696:4696+171], k=20)
#
# print('test')