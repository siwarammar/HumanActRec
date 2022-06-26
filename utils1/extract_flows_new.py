import os
import numpy as np
import cv2
from glob import glob
from multiprocessing import Pool


_IMAGE_SIZE = 256


def cal_for_frames(video_path):
    print(video_path)
    frames = glob(os.path.join(video_path, '*.jpg'))
    frames.sort()

    flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames):
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr

    return flow


def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()  #cv2.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2*bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow


def save_flow(video_flows, flow_path):
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path.format('u'), "TVL1jpg_y_{:06d}.jpg".format(i)),
                    flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path.format('v'), "TVL1jpg_y_{:06d}.jpg".format(i)),
                    flow[:, :, 1])


def gen_video_path():
    path = []
    flow_path = []
    length = []
    base = ''
    flow_base = ''
    for cls in os.listdir(base):
        videos = os.listdir(os.path.join(base, cls))
        for video in videos:
            tmp_path = os.path.join(base, cls, video)
            tmp_flow = os.path.join(flow_base, cls, '{:s}', video)
            tmp_len = len(glob(os.path.join(tmp_path, '*.jpg')))
            u = False
            v = False
            if os.path.exists(tmp_flow.format('u')):
                if len(glob(os.path.join(tmp_flow.format('u'), '*.jpg'))) == tmp_len:
                    u = True
            else:
                os.makedirs(tmp_flow.format('u'))
            if os.path.exists(tmp_flow.format('v')):
                if len(glob(os.path.join(tmp_flow.format('v'), '*.jpg'))) == tmp_len:
                    v = True
            else:
                os.makedirs(tmp_flow.format('v'))
            if u and v:
                print('skip:' + tmp_flow)
                continue

            path.append(tmp_path)
            flow_path.append(tmp_flow)
            length.append(tmp_len)
    return path, flow_path, length


def extract_flow(args):
    video_path = args[0]
    flow_video_path = args[1]
    print(video_path,flow_video_path )
    for video in os.listdir(video_path):
        flow_video_path_subclass = os.path.join(flow_video_path,video)
        if not os.path.exists(flow_video_path_subclass):
            os.mkdir(flow_video_path_subclass)

        flow = cal_for_frames(os.path.join(video_path,video))
        save_flow(flow, flow_video_path_subclass)
        print('complete:' + flow_video_path_subclass)
    return


if __name__ =='__main__':
    pool = Pool(2)   # multi-processing
    base_path = "/l/users/siwar.ammar/MARS/ucf_extracted_frames"
    flow_path = "/l/users/siwar.ammar/MARS/ucf_extracted_flows_new"
    #video_paths, flow_paths, video_lengths = gen_video_path()
    if not os.path.exists(flow_path):
            os.mkdir(flow_path)

    for cls in os.listdir(base_path):
        cls_flow_path = os.path.join(flow_path,cls)
        if not os.path.exists(cls_flow_path):
            os.mkdir(cls_flow_path)
        
        path = os.path.join(base_path, cls)
        extract_flow([path,cls_flow_path])


    