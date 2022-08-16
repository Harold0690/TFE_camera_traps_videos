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
    score0, score1, moy_p_0, moy_p_1, nbr1, p1 = 0, 0, 0, 0, 0, 0
    
    for out in list_output:   
        b = nn.functional.softmax(out, dim=1)
        prob, pred = torch.max(b, 1)
        preds.append(int(pred))
        probs.append(float(prob))
    
    if 1 in preds:
        final_pred = 1
    else:
        final_pred = 0
        
    if 0 in preds:
        nbr0 = preds.count(0)
        prob0 = [probs[i] for i,prd in enumerate(preds) if prd == 0]
        score0 = sum(prob0)/nbr0
        moy_p_0 = (math.exp(len(prob0)/(len(list_output)*10)))*score0
    
    if 1 in preds:
        nbr1 = preds.count(1)
        prob1 = [probs[i] for i,prd in enumerate(preds) if prd == 1]
        score1 = sum(prob1)/nbr1
        moy_p_1 = (math.exp(len(prob1)/(len(list_output)*10)))*score1
        
        p1 = nbr1 / len(list_output)
             
    return final_pred, score0, score1, moy_p_0, moy_p_1, preds, probs, p1

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
# dico = {'Athérure': 0, 'Autre': 1, 'Bongo': 2, 'Buffle de forêt': 3,
#         'Céphalophe à dos jaune': 4, 'Céphalophe bleu': 5, 'Céphalophe rouge': 6,
#         'Cercopithecidae': 7, 'Chevrotain aquatique': 8, 'Chimpanzé': 9,
#         'Civette/Genette/Nandinie': 10, 'Écureuil': 11, 'Eléphant de forêt': 12,
#         'Gorille': 13, 'Grand félin': 14, 'Mandrill': 15, 'Mangouste': 16,
#         'No_sp': 17, 'Oiseau': 18, 'Pangolin': 19, 'Potamochère': 20,
#         'Rat géant/Souris': 21}
dico = {'No_sp': 0, 'Animal': 1}

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

dataset = CSVVideoDataset(r"\\gf009pc059\Datafast\Harold\model_video\video_test20_2.csv",r"\\gf009pc059\Datafast\Harold\model_video\video_test20",2,trans)

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

model.load_state_dict(torch.load(r'\\gf009pc059\Datafast\Harold\model_video\resnet_model\resnet18_binaire_1.pt',map_location=torch.device('cpu')))

start4 = time.time()
print('Start validation bin simple : ',strftime('%H', gmtime(start4+7200)),'H',strftime('%M', gmtime(start4)),'min',strftime('%S', gmtime(start4)),'sec')

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
score0_csv = []
score1_csv = []
moy_p_0_csv = []
moy_p_1_csv = []
p1_csv = []

# track test loss
val_loss2 = 0.0
class_correct = list(0. for i in range(len(dico)))
class_total = list(0. for i in range(len(dico)))

cmpt1 = 0
cmpt2 = 0

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
            
        # print(type(data[0]))
        # print(len (data))
        # print(torch.unsqueeze(data[0],0).shape)
        # print(target)
        list_output = []
        list_pred = []
        for i in range(len(data)):
            output = model(torch.unsqueeze(data[i],0))
            list_output.append(output)
        
        final_pred, score0, score1, moy_p_0, moy_p_1, preds, probs, p1 = find_video_prediction(list_output)
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
        if train_on_gpu:
            pred = pred.cuda()
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
        y_pred.append(int(pred))
        y_true.append(int(target))
        # print(int(pred),int(target))
        
        # if int(target) != int(pred):
            # cmpt1 += 1
        target_csv.append(dico_inv.get(int(target)))
        pred_csv.append(dico_inv.get(int(pred)))
        names_csv.append(name)
        preds_csv.append(preds)
        probs_csv.append(probs)
        score0_csv.append(score0)
        score1_csv.append(score1)
        moy_p_0_csv.append(moy_p_0)
        moy_p_1_csv.append(moy_p_1)
        p1_csv.append(p1)
            # targets_csv.append((df_predictions_validation[df_predictions_validation['video']==name]['predictions'].values))
            # if dico_inv.get(int(target)) in preds2:
            #     cmpt2 += 1
            # shutil.copy(os.path.join(path_in,str(name[0])),os.path.join(path_error,str(classes[target])+"_"+str(classes[pred])+"_"+str(name[0])))

        
        # Incrementation of the progression bar (in a loop)
        printProgressBar(ti + 1, len(data_loader.dataset), prefix = 'Progress:', suffix = 'Complete', length = 50)

# print("Cas faux mais dont le bon label a été prédit au moins pour une frame de la vidéo : ", round((cmpt2/cmpt1)*100,2),' %')

df_results_valid = pd.DataFrame({'video': names_csv,'target':target_csv,'prediction':pred_csv,'score0':score0_csv,'score1':score1_csv,'moy_p_0':moy_p_0_csv,'moy_p_1':moy_p_1_csv,'preds':preds_csv,'probs':probs_csv,'p1':p1_csv})
df_results_valid.to_csv(r"\\gf009pc059\Datafast\Harold\model_video\results_test_simple_bin_fps2.csv",sep=";",index = False,encoding= 'latin-1')
       
mat_conf_val = confusion_matrix(y_true = y_true,y_pred = y_pred)
print(mat_conf_val)
# conf_val = numpy.zeros((len(dico)+1,len(dico)+2),dtype=object)
# conf_val[len(conf_val)-1,0] = 'total'
# for i in range(len(dico)):
#     conf_val[i,0] = classes[i]
#     for j in range(len(dico)):
#             conf_val[i,j+1] = mat_conf_val[i,j]
#             # print(mat_conf[i,j])
#             conf_val[i,len(dico)+1] += mat_conf_val[i,j]
#             conf_val[len(dico),len(dico)+1] += mat_conf_val[i,j]
#             conf_val[len(dico),j+1] += mat_conf_val[i,j]

# columns = []
# columns.append('target/predict')
# columns = columns + classes
# columns.append('total')

# df_cm_val = pd.DataFrame(conf_val,columns=columns)
# # print(df_cm_val)

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
