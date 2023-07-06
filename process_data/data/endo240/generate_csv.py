import os
# root = '/home/chenqingzhong/data/chenqingzhong/Data/CoCLR/endo_240p/frames/'
frame_root = 'E:/Download/duplicate_frames/'

dict = set()
for root, dirs, files in os.walk(frame_root):
    # 生成csv
    if len(files) != 0:
        print(root, '/,', len(files))
        # class_name = root.split('/')[-1].split('\\')[0]
        # video_name = root.split('/')[-1].split('\\')[1]
        # db_order = root.split('/')[-1]
        # print(db_order)
        # dict.add(db_order)
# print(dict)

# 生成ClassInd
# class_name = os.listdir(frame_root)
# print(class_name)