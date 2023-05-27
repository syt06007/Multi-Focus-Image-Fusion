from skimage import metrics
import cv2
import numpy as np
from PIL import Image
from glob import glob
from myTransforms import denorm


def metric(img_lst, mode):
    psnr_hist = []
    ssim_hist = []
    nmi_hist = []
    mean=[0.485, 0.456, 0.406] 
    std=[0.229, 0.224, 0.225]

    # model output에 사용할 때
    img_lst_d = [(denorm(mean,std, img).clamp(0,1)*255).numpy().astype('uint8') for img in img_lst]

    file_lst_gt = glob('/home/hyeongsik/dataset/LFDOF/test_data/ground_truth/*')
    file_lst_gt.sort()

    # file_lst = glob('ckpt_output/*')
    # file_lst.sort()

    for idx in range(1,len(file_lst_gt)):

        img1_gt = Image.open(file_lst_gt[idx])

        # img1 = Image.open(img_lst[idx].numpy())
        img1 = img_lst_d[idx]
        x = np.array(img1)
        x = x.transpose(1,2,0)
        y = np.array(img1_gt)

        y = y[:,:,:3]

        y = cv2.resize(y, [x.shape[1], x.shape[0]], interpolation=cv2.INTER_NEAREST)

        # print(x.shape, y.shape)
        if mode == 'PSNR':
            p = metrics.peak_signal_noise_ratio(x,y)
            psnr_hist.append(p)
            r = np.mean(psnr_hist)
        elif mode == 'SSIM':
            s = metrics.structural_similarity(x,y, multichannel=True)
            ssim_hist.append(s)
            r = np.mean(ssim_hist)
        elif mode == 'NMI':
            nmi = metrics.normalized_mutual_information(x,y)
            nmi_hist.append(nmi)
            r = np.mean(nmi_hist)

    return r