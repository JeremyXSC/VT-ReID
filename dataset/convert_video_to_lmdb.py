# -*- coding: utf-8 -*-
import os
import msgpack
import msgpack_numpy as m

m.patch()
from PIL import Image
import glob
import numpy as np
from tqdm import tqdm
from joblib import delayed, Parallel
import lmdb
from io import BytesIO
import random


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def pil_from_raw(raw):
    return Image.open(BytesIO(raw))


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def get_array(video_path):
    filelist = sorted(glob.glob(os.path.join(video_path, '*.jpg')))
    raw_list = list(map(lambda x: raw_reader(x), filelist))
    # array = np.stack(array_list, 0).astype(np.uint8)
    return raw_list


def put_action(action_path, txn, get_video_id):

    action_name = os.path.basename(action_path)
    video_list = sorted(glob.glob(os.path.join(action_path, '*')))
    name_list = [os.path.join(action_name, os.path.basename(vp)) for vp in video_list]
    array_list = Parallel(n_jobs=1)(delayed(get_array)(vp) for vp in tqdm(video_list, total=len(video_list)))

    vlen_list = [len(i) for i in array_list]
    for name, array in zip(name_list, array_list):
        vid = '%09d' % get_video_id[name]
        success = txn.put(vid.encode('ascii'), msgpack.dumps(array))
        if not success: print('%s failed to put in lmdb' % name)
    vid_list = ['%09d' % get_video_id[n] for n in name_list]
    return [i.encode('ascii') for i in vid_list], vlen_list


def make_dataset_lmdb(dataset_path, filename):
    lmdb_path = filename
    isdir = os.path.isdir(lmdb_path)
    print(f"Generate LMDB to {lmdb_path}")

    print(lmdb_path, isdir)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=int(2.9e10), readonly=False,
                   meminit=False,  map_async=True)
    txn = db.begin(write=True) #完整版使用4e10，部分版使用5e9

    # shuffle, assign id
    random.seed(0)
    video_list = sorted(glob.glob(os.path.join(dataset_path, '*', '*')))
    video_list = list(map(lambda x: '\\'.join(x.split('\\')[-2::]), video_list))
    random.shuffle(video_list)
    with open(lmdb_path + '-order', 'w') as f:
        f.write('\n'.join(video_list))
    get_video_id = dict(zip(video_list, range(len(video_list))))

    # start
    action_list = sorted(glob.glob(os.path.join(dataset_path, '*')))
    global_key_list = []
    video_len_list = []
    print(action_list)
    for i, ap in enumerate(action_list):
        print('[%d/%d]' % (i + 1, len(action_list)))
        key_list, vlen_list = put_action(ap, txn, get_video_id)
        global_key_list.append(key_list)
        video_len_list.extend(vlen_list)
        if i % 1 == 0:
            txn.commit()
            txn = db.begin(write=True)

    print("old video_list:", video_list)
    new_video_list = []
    for i in video_list:
        i = i.replace("\\", "/")
        new_video_list.append(i)
    video_list = new_video_list
    print("new video_list:", video_list)

    global_key_list = [k for sublist in global_key_list for k in sublist]
    txn.put(b'__keys__', msgpack.dumps(global_key_list))
    txn.put(b'__len__', msgpack.dumps(len(global_key_list)))
    txn.put(b'__order__', msgpack.dumps(video_list))
    txn.put(b'__vlen__', msgpack.dumps(video_len_list))

    txn.commit()
    print("Flushing database ...")
    db.sync()
    db.close()


import time


def read_lmdb(db_path):
    tic = time.time()
    env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                    readonly=True, lock=False,
                    readahead=False, meminit=False)
    print('Loading lmdb takes %.2f seconds' % (time.time() - tic))
    with env.begin(write=False) as txn:
        length = msgpack.loads(txn.get(b'__len__'))
        keys = msgpack.loads(txn.get(b'__keys__'))

    with env.begin(write=False) as txn:
        raw = msgpack.loads(txn.get(keys[0]))
    return raw

if __name__ == '__main__':
    # make_dataset_lmdb(dataset_path='/users/htd/beegfs/DATA/UCF101/frame',
    #                       filename='/users/htd/beegfs/DATA/UCF101/ucf101_frame.lmdb')
    # make_dataset_lmdb(dataset_path='/home/chenqingzhong/data/chenqingzhong/Data/CoCLR/endo_240p/frames',
    #                       filename='/home/chenqingzhong/data/chenqingzhong/Data/CoCLR/endo_240p/endo240p_frames_train.lmdb')
    # make_dataset_lmdb(dataset_path='E:/Download/frames',
    #                   filename='endo240p_frames_train')
    make_dataset_lmdb(dataset_path='D:\Colonpolyps\code\SCRL\data\Colo-SCRL\dataset',
                      filename='endo240p_duplicate_frames_train')
