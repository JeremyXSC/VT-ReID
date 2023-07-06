# import lmdb
# import msgpack
# import os
#
# def initialize(lmdb_dir):
#     env = lmdb.open(lmdb_dir) #如果没有就创建lmdb_dir目录
#     return env
#
# # lmdb_path = '/home/chenqingzhong/data/chenqingzhong/Data/CoCLR/UCF101/ucf101_frame.lmdb'
# # lmdb_path = '/home/chenqingzhong/data/chenqingzhong/Data/CoCLR/endo_240p/endo240p_video_train.lmdb'
# # lmdb_path = '/home/chenqingzhong/data/chenqingzhong/Data/CoCLR/endo_240/endo240p_frames_train.lmdb'
# lmdb_path = '/home/chenqingzhong/data/chenqingzhong/Data/CoCLR/endo_240p/endo240p_frames_train'
#
# env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
#                              readonly=True, lock=False,
#                              readahead=False, meminit=False)
# txn = env.begin(write=False)
# db_length = msgpack.loads(txn.get(b'__len__'), raw=True)
# db_keys = msgpack.loads(txn.get(b'__keys__'), raw=True)
# db_order = msgpack.loads(txn.get(b'__order__'), raw=True)
# get_video_id = dict(zip([i.decode() for i in db_order],
#                                      ['%09d'%i for i in range(len(db_order))]))
# cur = txn.cursor()
# for key, value in cur:
#     print(key, value)

list = ['wdm_1_1\\wdm_1_1_0397_0424', 'sxx_1_1\\sxx_1_1_0232_0247']
new_list = []
for i in list:
    print(i)
    i = i.replace("\\", "/")
    print(i)
    new_list.append(i)
print(new_list)