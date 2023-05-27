import torch
from torchvision import transforms, datasets
import os
from PIL import Image
import time

class Img_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.tf = transform

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
        # start = time.time()
        label = Image.open(os.path.join(self.data_dir + '/ground_truth' , self.lst_label[index]))
        label = self.totensor(label)

        if label.size(0) == 4:
            label = label[:3]

        input_path = os.path.join(self.data_dir + '/input')
        input_dir_list = os.listdir(os.path.join(input_path, self.lst_input[index]))
        input_lst = []
        
        for i in range(len(input_dir_list)):
            input_img = Image.open(os.path.join(input_path, self.lst_input[index], input_dir_list[i]))
            input_img =self.totensor(input_img)

            if input_img.size(0) == 4:
                input_img = input_img[:3]
            input_lst.append(input_img)

        stacked_input = torch.stack(input_lst, dim=0)
        label = label.unsqueeze(dim=0)
        cat_data = torch.cat((label, stacked_input), dim=0)
        
        cat_data = self.tf(cat_data)
        splited_data = torch.split(cat_data, split_size_or_sections=1, dim=0) # 쪼개기
        data_lst = [input.squeeze(dim=0) for input in splited_data] # 쪼갠걸 리스트에 요소로 넣기(모델이 다른게 다 리스트 기준으로 짜여져서 이게 편함)

        label = data_lst[0]
        input = data_lst[1:]


        data = {'input' : input, 'label' : label}
        # print(f"{time.time()-start:.4f}")

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
        
        data = {'input' : input} # 기존 코드 수정을 피하기 위해 dict 형태 유지
        # data : dict
        # input : tensor in list
        # label : tensor
        return data
