import torch

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import os

from myTransforms import denorm
from save_load import save, load
from dataset import test_Dataset

from model import LFFCNN
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Machine is :', device)

def predict(test_loader, model):
    model.eval()
    output_lst = []
    tti = transforms.ToPILImage()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            input = [img.to(device) for img in data['input']]

            output = model(input)

            output = output.cpu()
            output = output.squeeze(0)
            output_lst.append(output)

            file_name = os.path.join(folder_name, str(idx).zfill(2))
            img = denorm(mean, std, output).clamp(0,1)*255
            img = img.numpy().astype('uint8')
            img_save = Image.fromarray(img.transpose([1,2,0]))
            img_save.save(file_name + '.png', 'png')


if __name__ == '__main__':

    test_data_dir = './test_data'
    ckpt_dir = './checkpoint'
    folder_name = 'output'

    mean=[0.485, 0.456, 0.406] 
    std=[0.229, 0.224, 0.225]

    model = LFFCNN().to(device)

    model, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=model)
    
    transforms_compose = transforms.Compose([transforms.Resize((684,1008)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    dataset_test = test_Dataset(data_dir = test_data_dir, transform = transforms_compose)
    test_loader = DataLoader(dataset_test, batch_size = 1, shuffle = False, num_workers = 0)

    print('total iteration :', len(dataset_test))
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    predict(test_loader = test_loader, model = model)