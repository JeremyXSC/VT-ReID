import os
import shutil

root = '/home/chenqingzhong/data/chenqingzhong/Data/CoCLR/endo_240p/frames/'
class_root = os.path.join(root, '0class')

olddir_list = os.listdir(class_root)

for olddir_id in olddir_list:
    olddir = os.path.join(class_root, olddir_id)
    newdir_qianzhui = olddir_id.split('_')[:3]
    newdir_class = newdir_qianzhui[0] + '_' + newdir_qianzhui[1] + '_' + newdir_qianzhui[2]
    newdir = os.path.join(root, newdir_class, olddir_id)
    shutil.move(olddir, newdir)

# 打印片段数排查重复视频片段帧（有些息肉在同一分段）
# root = '/home/chenqingzhong/data/chenqingzhong/Data/CoCLR/endo_240p/frames/'
#
# olddir_list = os.listdir(root)
# olddir_list.sort()
#
# for olddir_id in olddir_list:
#     print(olddir_id)
#     olddir = os.path.join(root, olddir_id)
#     clip_list = os.listdir(olddir)
#     print(len(clip_list))