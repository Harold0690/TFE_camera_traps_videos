#%%
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
from typing import Any, Dict, List, Optional, Tuple
import time
from time import strftime,gmtime
import torch.nn as nn
from albumentations import SmallestMaxSize,PadIfNeeded,Normalize
from decord import VideoReader
from decord import cpu
from sklearn.metrics import confusion_matrix
import math

#%%
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
    
        indexes = [i for i, x in enumerate(compt) if x == max(compt)]
    

    if len(unique_preds) == 1:
        final_pred = int(unique_preds[0])
    # elif len(indexes) == 1:
    #     final_pred = int(unique_preds[indexes[0]])
    else:
        # new_probs = [unique_probs[i] for i in indexes]
        # new_unique_preds = [unique_preds[i] for i in indexes]
        # max_probs = [max(pb) for pb in new_probs]
        # indexes2 = [i for i, x in enumerate(max_probs) if x == max(max_probs)]
        # if len(indexes2) == 1:
        #     final_pred = int(new_unique_preds[indexes2[0]])
        # else:
        #     final_pred = [new_unique_preds[i] for i in indexes2]
        final_pred = int(unique_preds[moy_probs.index(max(moy_probs))])
            
    return final_pred, preds2, probs2

#%%
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
        csv_file: str, 
        root_dir: str,
        fps: int, 
        albu_transforms: Optional[list] = None
        ) -> None:
        ''' 
        Args:
            csv_file (str): absolute path to the csv file containing 
                videos names and labels
            root_dir (str) : path to the videos folder
            fps (int) : number of frames per second to extract from the video
            albu_transforms (list, optional): an albumentations' transformations 
                list that takes input sample as entry and returns a transformed 
                version. Defaults to None.
        '''

        assert isinstance(albu_transforms, (list, type(None))), \
            f'albumentations-transformations must be a list, got {type(albu_transforms)}'

        self.csv_file = csv_file
        self.root_dir = root_dir
        self.fps = fps
        self.df = pd.read_csv(csv_file,delimiter=';',encoding='latin-1')
        self.albu_transforms = albu_transforms

        self._img_names = numpy.unique(self.df['images'])
        self.labels = numpy.unique(self.df['labels'])
        self.labels.sort()
        self.dico_labels = {k:v for k,v in enumerate(self.labels)}

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

        # images.reverse()

        del self.images_dict["image0"]

        return images

    def _load_target(self, index: int) -> Dict[str,List[Any]]:
        img_name = self._img_names[index]
        annotations = self.df[self.df['images'] == img_name]
        annotations = annotations.drop(columns='images')

        annotations = list(annotations['labels'])
        target = numpy.where(self.labels == annotations[0])[0][0]

        return target

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
        target = self._load_target(index)

        tr_img = self._transforms(img)
        
        img_name = self._img_names[index]

        return tr_img, target, img_name
    
    def __len__(self) -> int:
        return len(self._img_names)
    
#%%
dico = {'Athérure': 0, 'Autre': 1, 'Bongo': 2, 'Buffle de forêt': 3,
        'Céphalophe à dos jaune': 4, 'Céphalophe bleu': 5, 'Céphalophe rouge': 6,
        'Cercopithecidae': 7, 'Chevrotain aquatique': 8, 'Chimpanzé': 9,
        'Civette/Genette/Nandinie': 10, 'Écureuil': 11, 'Eléphant de forêt': 12,
        'Gorille': 13, 'Grand félin': 14, 'Mandrill': 15, 'Mangouste': 16,
        'No_sp': 17, 'Oiseau': 18, 'Pangolin': 19, 'Potamochère': 20,
        'Rat géant/Souris': 21}

dico_inv = {v:k for k,v in dico.items()}

#%%
# create a complete CNN
model = models.resnet18(pretrained=True)
num_backbone_features = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(p=0.5),
                         nn.Linear(num_backbone_features, 256, bias=True),
                         nn.Dropout(p=0.5),
                         nn.Linear(256, len(dico), bias=True))
# print(model)

#%%
set_seed(6)

start = time.time()

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    
#%%

trans = [Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True, p=1.0),
         SmallestMaxSize(max_size=1080, interpolation=1, always_apply=True, p=1),
         PadIfNeeded (min_height=1080, min_width=1920, border_mode=cv.BORDER_CONSTANT, value=[0,0,0], always_apply=True, p=1.0),
         ToTensorV2()]

dataset = CSVVideoDataset(r"D:\Harold_TFE\video_valid20.csv",r"D:\Harold_TFE\video_valid20",1,trans)

nbr_vid = len(numpy.unique(dataset.df['images']))
indexes = [i for i in range(nbr_vid)]

sampler = SubsetRandomSampler(indexes)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler)

classes = [value for key,value in dico_inv.items()]

if train_on_gpu:
    model.cuda()

#%%
set_seed(6)

    ##################  
    # test the model #
    ##################

model.load_state_dict(torch.load(r'D:\Harold_TFE\resnet18_9.pt',map_location=torch.device('cpu')))

start4 = time.time()
print('Start validation moyenne pondérée 10 fpd = 0.5: ',strftime('%H', gmtime(start4+7200)),'H',strftime('%M', gmtime(start4)),'min',strftime('%S', gmtime(start4)),'sec')

# path_error = r'D:\Harold\resnet\error_validation'
# if os.path.exists(path_error):
#     shutil.rmtree(path_error)
#     os.makedirs(path_error)
# else:
#     os.makedirs(path_error)

# df_predictions_validation = pd.read_csv(r"D:\Harold_TFE\predictions_validation.csv",delimiter=';',encoding='latin-1')

y_pred = []
y_true = []
target_csv = []
pred_csv = []
names_csv = []
preds_csv = []
probs_csv = []
targets_csv = []

# track test loss
val_loss2 = 0.0
class_correct = list(0. for i in range(len(dico)))
class_total = list(0. for i in range(len(dico)))

cmpt1 = 0
cmpt2 = 0

names10 = []
y_pred1 = []

df1 = pd.read_csv(r"D:\Harold_TFE\Mbaza.csv",delimiter=';',encoding='latin-1')
df4 = pd.read_csv(r"D:\Harold_TFE\video_last_valid.csv",delimiter=';',encoding='latin-1')
path_in = r"D:\Harold_TFE\video_last_valid"
names1 = os.listdir(path_in)
names2 = [nms[:-4] for nms in names1]
scores3 =[]
for nms in names2:
    df2 = df1[df1["names"] == nms]
    indexes = df2.index
    scores2 = []
    for i in indexes:
        scores = [0 for h in range(22)]
        scores[df2['lab1'][i]] = df2['score_1'][i]
        scores[df2['lab2'][i]] = df2['score_2'][i]
        scores[df2['lab3'][i]] = df2['score_3'][i]
        scores2.append(torch.unsqueeze(torch.tensor(scores),0))
    scores3.append(scores2)

df3 = pd.DataFrame({'video': names1,'score':scores3})

model.eval()
# # Initialisation of the progression bar
# printProgressBar(0, len(data_loader.dataset), prefix = 'Progress:', suffix = 'Complete', length = 50)

# Initialisation of the progression bar
printProgressBar(0, len(df3), prefix = 'Progress:', suffix = 'Complete', length = 50)

# iterate over test data
with torch.no_grad():
    # for ti,(data, target, name) in enumerate(data_loader):
    for hi in range(len(df3)):
        # data = data[0]
        # name = name[0]
        # # move tensors to GPU if CUDA is available
        # if train_on_gpu:
        #     data, target = data.cuda(), target.cuda()
            
        # # print(type(data[0]))
        # # print(len (data))
        # # print(torch.unsqueeze(data[0],0).shape)
        # # print(target)
        # list_output = []
        # list_pred = []
        # for h in range(len(data)):
        #     output = model(torch.unsqueeze(data[i],0))
        #     list_output.append(output)
        
        # print(list_output)
        list_output = list(df3['score'][hi])
        target = int((df4[df4['images'] == df3['video'][hi]]['labels']))
        target = torch.tensor(target)
        name = df3['video'][hi]
        names10.append(name)
        
        final_pred, preds, probs = find_video_prediction(list_output)
        y_pred1.append(final_pred)
        # print(final_pred)
        # print(type(final_pred))
        if type(final_pred) == int:
            pred = torch.tensor([final_pred])
        else:
            # print(type(final_pred))
            # print(final_pred)
            pred = torch.tensor([final_pred[0]])
            print('\nTarget : ',int(target)," | Final prediction : ", final_pred)
            # print(output)
            # _, pred = torch.max(output, 1) 
            # list_pred.append(pred)
        # print(list_pred)
        # print(max(set(list_pred), key = list_pred.count)) # 3
            
        # forward pass: compute predicted outputs by passing inputs to the model
        # output7 = model(torch.unsqueeze(data[0],0))
        # calculate the batch loss
        # if train_on_gpu:
        #     target = target.cpu()
        #     output = output.cpu()
        # loss = criterion(output, target)
        # update test loss 
        # val_loss2 += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        # _, pred = torch.max(output, 1)
        # print(pred)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = numpy.squeeze(correct_tensor.numpy()) if not train_on_gpu else numpy.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        label = target.data
        class_correct[label] += correct.item()
        class_total[label] += 1
        
        # output5 = nn.functional.softmax(output, dim=1)
        # output6 = (output5[0]).tolist()
        # output7 = [round(i,3) for i in output6]
        # print("\n===================================================================================")
        # print('Output : ',output7)
        # print("Target : ",classes[target]," | Prediction : ",classes[pred]," | Probability : ",round(((output5[0]).tolist())[pred]*100,3)," %")      
        # print("===================================================================================")
        # names10.append(name)
        y_pred.append(int(pred))
        y_true.append(int(target))
        # print(int(pred),int(target))
        
        if int(target) != int(pred):
            cmpt1 += 1
            target_csv.append(dico_inv.get(int(target)))
            pred_csv.append(dico_inv.get(int(pred)))
            names_csv.append(name)
            preds2 = [dico_inv.get(i) for i in preds]
            preds_csv.append(preds2)
            probs_csv.append(probs)
            # targets_csv.append((df_predictions_validation[df_predictions_validation['video']==name]['predictions'].values))
            if dico_inv.get(int(target)) in preds2:
                cmpt2 += 1
            # shutil.copy(os.path.join(path_in,str(name[0])),os.path.join(path_error,str(classes[target])+"_"+str(classes[pred])+"_"+str(name[0])))

        
        # Incrementation of the progression bar (in a loop)
        printProgressBar(hi + 1, len(df3), prefix = 'Progress:', suffix = 'Complete', length = 50)

print("Cas faux mais dont le bon label a été prédit au moins pour une frame de la vidéo : ", round((cmpt2/cmpt1)*100,2),' %')

df_results_valid = pd.DataFrame({'video': names_csv,'target':target_csv,'prediction':pred_csv,'predictions':preds_csv,'probabilities':probs_csv})
df_results_valid.to_csv(r"D:\Harold_TFE\results_validation_moy_pond.csv",sep=";",index = False,encoding= 'latin-1')

df_res18 = pd.DataFrame({'images': names10,'labels':y_pred1})
df_res18.to_csv(r"D:\Harold_TFE\Mbaza_top1.csv",sep=";",index = False,encoding= 'latin-1')
        
mat_conf_val = confusion_matrix(y_true = y_true,y_pred = y_pred)
conf_val = numpy.zeros((len(dico)+1,len(dico)+2),dtype=object)
conf_val[len(conf_val)-1,0] = 'total'
for i in range(len(dico)):
    conf_val[i,0] = classes[i]
    for j in range(len(dico)):
            conf_val[i,j+1] = mat_conf_val[i,j]
            # print(mat_conf[i,j])
            conf_val[i,len(dico)+1] += mat_conf_val[i,j]
            conf_val[len(dico),len(dico)+1] += mat_conf_val[i,j]
            conf_val[len(dico),j+1] += mat_conf_val[i,j]

columns = []
columns.append('target/predict')
columns = columns + classes
columns.append('total')

df_cm_val = pd.DataFrame(conf_val,columns=columns)
# print(df_cm_val)

vector_accuracy_val = numpy.zeros((1,len(dico)))

for i in range(len(dico)):
    if class_total[i] > 0:
        print('Validation Accuracy of %5s: %2d%% (%2d/%2d)' % (
            dico_inv.get(i), 100 * class_correct[i] / class_total[i],
            numpy.sum(class_correct[i]), numpy.sum(class_total[i])))
        vector_accuracy_val[0,i] = round(100 * class_correct[i] / class_total[i],2)
    else:
        print('Validation Accuracy of %5s: N/A (no training examples)' % (classes))

print('\nValidation Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * numpy.sum(class_correct) / numpy.sum(class_total),
    numpy.sum(class_correct), numpy.sum(class_total)))

global_acc_val = 100. * numpy.sum(class_correct) / numpy.sum(class_total)

temps4 = time.time() - start4
print(" ")
print("Temps écoulé pour la validation :",strftime('%H', gmtime(temps4)),'H',strftime('%M', gmtime(temps4)),'min',strftime('%S', gmtime(temps4)),'sec')
