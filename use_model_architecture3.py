#%%
# Import libraries
import torch
import os
import numpy
import albumentations
import cv2 as cv
import pandas as pd
import random
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from typing import Optional, Tuple
import time
from time import strftime,gmtime
import torch.nn as nn
from albumentations import SmallestMaxSize,PadIfNeeded,Normalize
from decord import VideoReader
from decord import cpu
import math

#%%
# Definitions
def set_seed(seed):
    ''' 
    Function to set seed for reproducibility 

    Perfect reproducibility is not guaranteed
    see https://pytorch.org/docs/stable/notes/randomness.html
    '''
    # CPU variables
    random.seed(seed) 
    numpy.random.seed(seed)
    # Python
    torch.manual_seed(seed) 
    # GPU variables
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    # torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False

#%%
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()
#%%
#TOP1
def find_video_prediction(list_output):
    
    preds = []
    probs = []
    moy_probs = []
    
    for out in list_output:   
        b=out
        # b = nn.functional.softmax(out, dim=1)
        prob, pred = torch.max(b, 1)
        preds.append(int(pred))
        probs.append(float(prob))
    
    compt = []
    unique_probs = []
    unique_preds = list(numpy.unique(preds))
    
    preds2 = preds.copy()
    probs2 = probs.copy()
    
    if len(unique_preds) != 1:
        if 17 in unique_preds:
            unique_preds.remove(17)
            indexes17 = [i for i, x in enumerate(preds) if x == 17]
            indexes17.sort(reverse=True)
            for i in indexes17:
                preds.pop(i)
                probs.pop(i)
        for prd in unique_preds:
            compt.append(preds.count(prd))
            unique_probs.append([probs[i] for i, x in enumerate(preds) if x == prd])
            porbs4 = [probs[i] for i, x in enumerate(preds) if x == prd]
            moy = sum(porbs4)/len(porbs4)
            moy_probs.append((math.exp(len(porbs4)/(len(list_output)*10)))*moy)
    
    if len(unique_preds) == 1:
        final_pred = int(unique_preds[0])
    else:
        final_pred = int(unique_preds[moy_probs.index(max(moy_probs))])
            
    return final_pred, preds2, probs2

#%%
# TOP3
def find_video_prediction_top3(list_output):
    
    score = [0 for i in range(22)]
    lab =  [i for i in range(22)]
    pred12 = []
    prob12 = []

    for out in list_output:
        b = nn.functional.softmax(out, dim=1)
        prob10,pred10 = torch.topk(b, k = 3, dim = 1)
        pred11 = pred10.tolist()[0]
        prob11 = prob10.tolist()[0]
        pred12.append([dico_inv.get(i) for i in pred11])
        prob12.append(prob10.tolist())
        for i,cl in enumerate(pred11):
            # print(cl)
            score[cl] += prob11[i]
    
    score2 = [score[i] / len(list_output) for i in range(len(score))]
    score17 = round(score2[17],2)
    a = sorted(zip(score2, lab), reverse=True)
    if a[2][1] == 17 and a[2][0] < 0.5:
        a.pop(2)
    elif a[1][1] == 17 and a[1][0] < 0.5:
        a.pop(1)
    elif a[0][1] == 17 and a[0][0] < 0.5:
        a.pop(0)
    
    final_preds3 = list([a[0][1],a[1][1],a[2][1]])
    scores = list([a[0][0]*len(list_output),a[1][0]*len(list_output),a[2][0]*len(list_output)])
    
    return final_preds3, scores, score, pred12, prob12, score17

#%%
# Class dataset
class CSVVideoDataset(Dataset):
    ''' Class to create a Dataset from a CSV file containing video data's 
    
    This dataset is built on the basis of CSV files. The CSV files have to contain a 
    column named ['images'] in which the name of the videos is written. They also contain
    a ['labels'] column with the labels of the videos. One line represent one label for 
    one video. If there are several labels for only one video, there are several lines in
    the CSV files for this video.
    Return the frames of a video in a list or a tensor (it depends on the transformations) 
    and the target. The target is a numpy array filled with several 0.0 or 1.0 that
    corrospond to the labels of the video.
    '''

    def __init__(
        self, 
        root_dir: str,
        fps: int, 
        albu_transforms: Optional[list] = None
        ) -> None:
        ''' 
        Args:
            root_dir (str) : path to the videos folder
            fps (int) : number of frames per second to extract from the video
            albu_transforms (list, optional): an albumentations' transformations 
                list that takes input sample as entry and returns a transformed 
                version. Defaults to None.
        '''

        assert isinstance(albu_transforms, (list, type(None))), \
            f'albumentations-transformations must be a list, got {type(albu_transforms)}'

        self.root_dir = root_dir
        self.fps = fps
        self.albu_transforms = albu_transforms

        self._img_names = os.listdir(self.root_dir)

    def _load_image(self, index: int):
        img_name = self._img_names[index]
        img_path = os.path.join(self.root_dir, img_name)
        vr = VideoReader(img_path, ctx=cpu(0))
        cap = cv.VideoCapture(img_path)
        frames_per_second = (cap.get(cv.CAP_PROP_FPS))
        images = []
        self.images_dict = {}
        frames_list = list(range(0, len(vr), int(frames_per_second/self.fps)))
        for i,idx in enumerate(frames_list):
            images.append(vr[idx].asnumpy())
            self.images_dict["image"+str(i)] = 'image'

        del self.images_dict["image0"]

        return images

    def _transforms(
        self, 
        images: list
        ) -> Tuple[torch.Tensor]:

        if self.albu_transforms:
            transform_pipeline = albumentations.Compose(self.albu_transforms,additional_targets=self.images_dict)
            images2 = images.copy()
            images2.pop(0)
            kwargs2 = {k:img for (k,v),img in zip(self.images_dict.items(),images2)}
            tr_image = transform_pipeline(image=images[0],**kwargs2)
            tr_image_list = [v for k,v in tr_image.items()]
            unsqueezed_img_list = torch.stack(tr_image_list)

            return unsqueezed_img_list
        
        else:
            print('No transformation')
            return images

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, list]:        
        img = self._load_image(index)

        tr_img = self._transforms(img)
        
        img_name = self._img_names[index]

        return tr_img, img_name
    
    def __len__(self) -> int:
        return len(self._img_names)
    
#%%
dico = {'Porcupine_brush_tailed': 0, 'Other': 1, 'Bongo': 2, 'Buffalo_african': 3,
        'Duiker_yellow_backed': 4, 'Duiker_blue': 5, 'Duiker_red': 6,
        'Cercopithecid': 7, 'Chevrotain_Water': 8, 'Chimpanzee': 9,
        'Civet_Genet_Nandinia': 10, 'Squirrel': 11, 'Elephant_african': 12,
        'Gorilla': 13, 'Cat_golden_Leopard_african': 14, 'Mandrill': 15, 'Mongoose': 16,
        'No_sp': 17, 'Bird': 18, 'Pangolin': 19, 'Hog_red_river': 20,
        'Rat_giant_Mouse': 21}

dico_inv = {v:k for k,v in dico.items()}

#%%
# create a complete CNN
model = models.resnet18(pretrained=True)
num_backbone_features = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(p=0.5),
                         nn.Linear(num_backbone_features, 256, bias=True),
                         nn.Dropout(p=0.5),
                         nn.Linear(256, len(dico), bias=True))

#%%
# Hyperparameters
batch_size = 1
H = 1080
L = 1920
seed = 100

##########
# INPUTS #  
##########
# (TO MODIFY)

model_path = r'C:\resnet18_9.pt'
path_in = r"C:\video"
path_csv_out =  r"C:\out.csv"


#%%
set_seed(seed)

start = time.time()

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    
#%%

trans = [Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True, p=1.0),
         SmallestMaxSize(max_size=H, interpolation=1, always_apply=True, p=1),
         PadIfNeeded (min_height=H, min_width=L, border_mode=cv.BORDER_CONSTANT, value=[0,0,0], always_apply=True, p=1.0),
         ToTensorV2()]

dataset = CSVVideoDataset(path_in,1,trans)

nbr_vid = len(os.listdir(path_in))
indexes = [i for i in range(nbr_vid)]

sampler = SubsetRandomSampler(indexes)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)

classes = [value for key,value in dico_inv.items()]

if train_on_gpu:
    model.cuda()

#%%
set_seed(seed)

    #################  
    # Use the model #
    #################

model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

start4 = time.time()
print('Start : ',strftime('%H', gmtime(start4+7200)),'H',strftime('%M', gmtime(start4)),'min',strftime('%S', gmtime(start4)),'sec')

y_pred = []
y_pred1 = []
y_pred2 = []
y_pred3 = []
names_csv = []

model.eval()
# Initialisation of the progression bar
printProgressBar(0, len(data_loader.dataset), prefix = 'Progress:', suffix = 'Complete', length = 50)

# iterate over test data
with torch.no_grad():
    for ti,(data, target, name) in enumerate(data_loader):
        data = data[0]
        name = name[0]
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        list_output = []
        list_pred = []
        for i in range(len(data)):
            output = model(torch.unsqueeze(data[i],0))
            list_output.append(output)
        
        final_pred, preds, probs = find_video_prediction(list_output)
        final_preds, scores, score, pred12, prob12, score17 = find_video_prediction_top3(list_output)
        names_csv.append(name)
        y_pred.append(final_pred)
        y_pred1.append(final_preds[0])
        y_pred2.append(final_preds[1])
        y_pred3.append(final_preds[2])
        
        # Incrementation of the progression bar (in a loop)
        printProgressBar(ti + 1, len(data_loader.dataset), prefix = 'Progress:', suffix = 'Complete', length = 50)

df_results = pd.DataFrame({'video': names_csv,'prediction_top1':y_pred,'prediction_1_top3':y_pred1,'prediction_2_top3':y_pred2,'prediction_3_top3':y_pred3})
df_results.to_csv(path_csv_out,sep=";",index = False,encoding= 'latin-1')
