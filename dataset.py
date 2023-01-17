import torch
from torchvision import transforms, datasets
import os
from PIL import Image

class Img_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform = None, rotate = None):
        self.data_dir = data_dir
        self.tf = transform
        self.rotate = rotate
        self.totensor = transforms.ToTensor()

        lst_label = os.listdir(os.path.join(data_dir, 'ground_truth'))
        lst_input = os.listdir(os.path.join(data_dir, 'input'))

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = Image.open(os.path.join(self.data_dir + '/ground_truth' , self.lst_label[index]))

        if self.rotate: 
            label = label.rotate(self.rotate)

        label = self.totensor(label)

        if label.size(0) == 4:
            label = label[:3]

        label = self.tf(label)

        input_path = os.path.join(self.data_dir + '/input')
        input_dir_list = os.listdir(os.path.join(input_path, self.lst_input[index]))

        input = []
        
        for i in range(len(input_dir_list)):
            input_img = Image.open(os.path.join(input_path, self.lst_input[index], input_dir_list[i]))
            
            if self.rotate:
                input_img = input_img.rotate(self.rotate)

            input_img =self.totensor(input_img)

            if input_img.size(0) == 4:
                input_img = input_img[:3]

            input.append(self.tf(input_img))

            # if input[i].size(0) == 4:
            #     input[i] = input[i][:3]
        
        data = {'input' : input, 'label' : label} 
        # data : dict
        # input : tensor in list
        # label : tensor
        return data

class test_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform = None, rotate = None):
        self.data_dir = data_dir
        self.tf = transform
        
        lst_input = os.listdir(self.data_dir)
        lst_input.sort()
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_input)

    def __getitem__(self, index):

        input_path = self.data_dir
        input_dir_list = os.listdir(os.path.join(input_path, self.lst_input[index]))

        input = []
        
        for i in range(len(input_dir_list)):
            input_img = Image.open(os.path.join(input_path, self.lst_input[index], input_dir_list[i]))

            input.append(self.tf(input_img))
            
            if input[i].size(0) == 4:
                input[i] = input[i][:3]
        
        data = {'input' : input} 
        # data : dict
        # input : tensor in list
        # label : tensor
        return data
