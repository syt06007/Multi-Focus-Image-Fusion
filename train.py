import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from myTransforms import denorm
from PIL import Image

# from ignite.metrics import *
from tqdm import tqdm
import time
import numpy as np
import os

from save_load import save, load

from dataset import Img_dataset
# from models.model_LFFCNN import LFFCNN
from models.model_spp import LFFCNN
# from models.IFCNN import myIFCNN
from resnet101_perceptual_loss_khs import ResNetPerceptualLoss
from img_metric import metric

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print('Machine is',device)

def train(ckpt_dir, val_loader, st_epoch, epochs, model, loss_fn1, loss_fn2, optim):
    writer = SummaryWriter()
    train_data1 = '/home/hyeongsik/dataset/LFDOF/train_data'
    train_data2 = '/home/hyeongsik/dataset/LFDOF2'

    train_dataset = Img_dataset(data_dir = train_data1, 
                                transform= transforms.Compose([transforms.Resize((342, 504)), transforms.Normalize(mean,std),
                                        transforms.RandomVerticalFlip(p=0.5), transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomRotation(90)])) # 546,804
    train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = True, num_workers=8)
    
    best_metric_value = 0.0
    for epoch in range(st_epoch +1, epochs):
        train_loss_hist = 0.0
        idx = 0
        print("[ EPOCH ", epoch, " ]")

        train_loss_hist, num_idx = trainer(train_loader, model, loss_fn1, loss_fn2)
                
        mean_train_loss = train_loss_hist / float(num_idx)

        # validation 추가
        mean_val_loss, output_lst, metric_value = validation(model, val_loader, loss_fn1, loss_fn2)
        path = './best_model/'
        if metric_value > best_metric_value : 
            torch.save(model.state_dict(), path + 'model_state_dict_SPP.pt')
            best_metric_value = metric_value
            print('best model!', best_metric_value)


        writer.add_scalar('Loss / Train', mean_train_loss, epoch)
        writer.add_scalar('Loss / Validation', mean_val_loss, epoch)
        writer.add_scalar('Metric', metric_value, epoch)

        writer.add_image('Val IMG/1', output_lst[1], epoch)
        writer.add_image('Val IMG/2', output_lst[2], epoch)
        writer.add_image('Val IMG/3', output_lst[3], epoch)
        writer.add_image('Val IMG/4', output_lst[4], epoch)
        
        if epoch % 10 == 0:
            save(ckpt_dir=ckpt_dir, net=model, optim=optim, epoch=epoch)

def trainer(train_loader, model, loss_fn1, loss_fn2):
    model.train()
    loss_hist = 0.0
    w1 = 0.3
    w2 = 0.7

    for idx, data in enumerate(tqdm(train_loader)): 
        label = data['label'].to(device)
        input = [img.to(device) for img in data['input']]
        output = model(input)

        optim.zero_grad()
        loss = w1 * loss_fn1(output, label) + w2 * loss_fn2(output, label)
        loss.backward()
        optim.step()
        loss_hist += loss.item()

    loss_hist = loss_hist/float(idx)
    torch.cuda.empty_cache()
   
    return loss_hist, idx

def validation(model, val_loader, loss_fn1, loss_fn2):
    model.eval()

    val_loss_hist = 0.0
    output_lst = []
    with torch.no_grad():
        for idx, data in enumerate(tqdm(val_loader)):
            label = data['label'].to(device)
            input = [img.to(device) for img in data['input']]

            output = model(input)
            loss = loss_fn1(output, label) + loss_fn2(output, label)
            val_loss_hist += loss.item()

            output = output.cpu()
            output = output.squeeze(0)
            output_lst.append(output)

        metric_value = metric(output_lst, mode = 'PSNR')

    val_loss = val_loss_hist / float(idx)
    print('Val LOSS : {loss} | PSNR : {metric_value}'.format(loss=loss, metric_value=metric_value))
    return val_loss, output_lst, metric_value

def predict(test_loader, model):
    model.eval()
    
    folder_name = 'ckpt_output'
    folder_name2 = 'ckpt_output_tti'

    output_lst = []
    tti = transforms.ToPILImage()
    with torch.no_grad():
        for idx, data in tqdm(enumerate(test_loader)):
            input = [img.to(device) for img in data['input']]

            output = model(input)

            output = output.cpu()
            output = output.squeeze(0)
            output_lst.append(output)

            file_name2 = os.path.join(folder_name2, str(idx).zfill(2))
            img = tti(output)
            img.save(file_name2 + '.png', 'png')

            file_name = os.path.join(folder_name, str(idx).zfill(2))
            img = denorm(mean, std, output).clamp(0,1)*255
            img = img.numpy().astype('uint8')
            img_save = Image.fromarray(img.transpose([1,2,0]))
            img_save.save(file_name + '.png', 'png')

if __name__ == '__main__':
    epochs = 200
    lr = 1e-3
    batch_size = 1
    st_epoch=0

    val_data_dir = '/home/hyeongsik/dataset/LFDOF2'
    test_data_dir = '/home/hyeongsik/dataset/LFDOF/test_data'
    ckpt_dir = './checkpoint'
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    
    mean=[0.485, 0.456, 0.406] 
    std=[0.229, 0.224, 0.225]

    val_dataset = Img_dataset(data_dir = test_data_dir,  transform= transforms.Compose([transforms.Resize((684, 1008)), transforms.Normalize(mean,std)]))
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False, num_workers=8)

    test_dataset = Img_dataset(data_dir = test_data_dir,  transform= transforms.Compose([transforms.Resize((684,1008)), transforms.Normalize(mean,std)]))
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers=8)

    model = LFFCNN().to(device)
    # model = myIFCNN().to(device)

    loss_fn1 = nn.MSELoss().to(device)
    loss_fn2 = ResNetPerceptualLoss().to(device)
    optim = torch.optim.Adam(model.parameters(), lr = lr)


    model, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=model, optim=optim)
    print("Start Epochs :", st_epoch)

    test_img_dir = './testimg'
    if not os.path.exists(test_img_dir):
        os.mkdir(test_img_dir)

    train(ckpt_dir = ckpt_dir, val_loader = val_loader, st_epoch=st_epoch, epochs=epochs, model=model, loss_fn1=loss_fn1, loss_fn2=loss_fn2, optim=optim)
    
    model.load_state_dict(torch.load('./best_model/model_state_dict.pt'))
    predict(model = model, test_loader = test_loader)