import os.path
from moviepy.editor import VideoFileClip

# 遍历其子文件夹
def get_lens(path):
    times = 0
    cnt = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            print(os.path.join(dirpath, dirname))
            files =  [os.path.join(dirpath, dirname, f) for f in os.listdir(os.path.join(dirpath, dirname)) if f.endswith('.mp4') or f.endswith('.avi')]

            for f in files:
                print(f)
                clip = VideoFileClip(f)
                # 计算视频的时长，单位为分钟
                len = f, round(clip.duration / 60, 2)
                times+=len[1]
                clip.close()
                cnt += 1
    print("分批视频时长:",times)
    print("分批视频数量:", cnt)

# 邱志斌：
# 陆宝龙1
# 陆宝龙2
# CJ030797_陈雪英
# CJ030756_顾敬华
# 许建峰1
# 许建峰2
# get_lens('/home/chenqingzhong/data/chenqingzhong/Data/zhongshan_endoscope/结肠息肉录像_第三批/')
# get_lens('/home/chenqingzhong/data/chenqingzhong/Data/zhongshan_endoscope/结肠息肉录像_第四批/')

# 遍历当前文件夹
# def get_lens(path):
#     times = 0
#     cnt = 0
#     files = [os.path.join(path, f) for f in os.listdir(os.path.join(path)) if
#              f.endswith('.mp4') or f.endswith('.avi')]
#     for f in files:
#         print(f)
#         clip = VideoFileClip(f)
#         # 计算视频的时长，单位为分钟
#         len = f, round(clip.duration / 60, 2)
#         times+=len[1]
#         clip.close()
#         cnt += 1
#     print("分批视频时长:",times)
#     print("分批视频数量:", cnt)
# get_lens('/home/chenqingzhong/data/chenqingzhong/Data/zhongshan_endoscope/结肠息肉录像_去黑边/')