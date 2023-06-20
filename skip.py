import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_crop
from util import html
import numpy as np
import math
from PIL import Image
import torchvision.transforms as transforms
import torch
import random
import cv2
import dlib
from skimage import transform as trans
from skimage import io
from data.image_folder import make_dataset
import sys
from tqdm import tqdm

sys.path.append('FaceLandmarkDetection')
import face_alignment


def AddUpSample(img):
    return img.resize((512, 512), Image.BICUBIC)


def get_part_location(Landmarks):
    Map_LE = list(np.hstack((range(17, 22), range(36, 42))))
    Map_RE = list(np.hstack((range(22, 27), range(42, 48))))
    Map_NO = list(range(29, 36))
    Map_MO = list(range(48, 68))
    try:
        # left eye
        Mean_LE = np.mean(Landmarks[Map_LE], 0)
        L_LE = np.max((np.max(np.max(Landmarks[Map_LE], 0) - np.min(Landmarks[Map_LE], 0)) / 2, 16))
        Location_LE = np.hstack((Mean_LE - L_LE + 1, Mean_LE + L_LE)).astype(int)
        # right eye
        Mean_RE = np.mean(Landmarks[Map_RE], 0)
        L_RE = np.max((np.max(np.max(Landmarks[Map_RE], 0) - np.min(Landmarks[Map_RE], 0)) / 2, 16))
        Location_RE = np.hstack((Mean_RE - L_RE + 1, Mean_RE + L_RE)).astype(int)
        # nose
        Mean_NO = np.mean(Landmarks[Map_NO], 0)
        L_NO = np.max((np.max(np.max(Landmarks[Map_NO], 0) - np.min(Landmarks[Map_NO], 0)) / 2, 16))
        Location_NO = np.hstack((Mean_NO - L_NO + 1, Mean_NO + L_NO)).astype(int)
        # mouth
        Mean_MO = np.mean(Landmarks[Map_MO], 0)
        L_MO = np.max((np.max(np.max(Landmarks[Map_MO], 0) - np.min(Landmarks[Map_MO], 0)) / 2, 16))
        Location_MO = np.hstack((Mean_MO - L_MO + 1, Mean_MO + L_MO)).astype(int)
    except:
        return 0
    return torch.from_numpy(Location_LE).unsqueeze(0), torch.from_numpy(Location_RE).unsqueeze(0), torch.from_numpy(
        Location_NO).unsqueeze(0), torch.from_numpy(Location_MO).unsqueeze(0)


def obtain_inputs(A_paths, Landmark_path):
    # A_paths = img_path
    A = Image.open(A_paths).convert('RGB')
    Part_locations = get_part_location(Landmark_path)
    if Part_locations == 0:
        return 0
    C = A
    A = AddUpSample(A)
    A = transforms.ToTensor()(A)
    C = transforms.ToTensor()(C)
    A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)  #
    C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C)  #
    return {'A': A.unsqueeze(0), 'C': C.unsqueeze(0), 'A_paths': A_paths, 'Part_locations': Part_locations}


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    opt.which_epoch = 'latest'  #

    dev = 'cuda:{}'.format(opt.gpu_ids[0])
    FD = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=dev, flip_input=False)
    model = create_model(opt)
    model.setup(opt)

    TestImgPath = opt.test_path
    true_output_path = opt.results_dir
    UpScaleWhole = opt.upscale_factor

    print('\n###################### init done ##############################')
    count = 0
    aim_list = list(os.walk(TestImgPath))
    random.shuffle(aim_list)
    for r, d, f in tqdm(aim_list):
        for ff in f:
            if ff.endswith('jpg'):
                ff_p = os.path.join(r, ff)
                ff_out_p = ff_p.replace(TestImgPath, true_output_path)
                if not os.path.exists(ff_out_p):
                    os.makedirs(r.replace(TestImgPath, true_output_path), exist_ok=True)

                    Img = io.imread(ff_p)
                    try:
                        PredsAll = FD.get_landmarks(Img)
                    except:
                        print('\t################ Error in face detection, continue...')
                        continue
                    if PredsAll is None:
                        print('\t################ No face, continue...')
                        continue
                    ins = 0
                    if len(PredsAll) != 1:
                        hights = []
                        for l in PredsAll:
                            hights.append(l[8, 1] - l[19, 1])
                        ins = hights.index(max(hights))
                    preds = PredsAll[ins]

                    # torch.cuda.empty_cache()
                    data = obtain_inputs(ff_p, preds[:, 0:2])
                    if data == 0:
                        print('\t################ Error in landmark file, continue...')
                        continue  #
                    model.set_input(data)
                    try:
                        model.test()
                        visuals = model.get_current_visuals()
                        save_crop(visuals, ff_out_p)
                    except Exception as e:
                        print('\t################ Error in enhancing this image: {}'.format(str(e)))
                        print('\t################ continue...')
                        continue

                    count += 1

        if count % 10000 == 0:
            print('already process {} imgs'.format(count))