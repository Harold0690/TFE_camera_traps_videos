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
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Tuple
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from time import strftime,gmtime
import torch.nn as nn
from albumentations import SmallestMaxSize,PadIfNeeded,Normalize
from decord import VideoReader
from decord import cpu
import torch.nn.functional as F
from torchvision.models import resnet18
from sklearn.metrics import confusion_matrix

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
# Class dataset
class CSVVideoDataset(Dataset):
    ''' Class to create a Dataset from a CSV file containing video data's 
    
    This dataset is built on the basis of CSV files. The CSV files have to contain a 
    column named ['videos'] in which the name of the videos is written. They also contain
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
        dico: dict,
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

        self._img_names = numpy.unique(self.df['videos'])
        self.labels = numpy.unique(self.df['labels'])
        self.labels.sort()
        self.dico_labels = dico
        
        
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
        annotations = self.df[self.df['videos'] == img_name]
        annotations = annotations.drop(columns='videos')

        annotations = list(annotations['labels'])
        target = annotations[0]

        return target, img_name

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
        target, img_name = self._load_target(index)

        tr_img = self._transforms(img)

        return tr_img, target, img_name
    
    def __len__(self) -> int:
        return len(self._img_names)

#%%
# Class LSTM
class CNNLSTM(nn.Module):
    def __init__(self, nbr_cl):
        super(CNNLSTM, self).__init__()
        
        resnet = resnet18(pretrained=True)
        resnet = list(resnet.children())
        self.resnet = nn.Sequential(*resnet[:-1])
        
        self.fc0 = nn.Sequential(nn.Flatten(),nn.Linear(512, 300))
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, nbr_cl)
       
    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])
            x = self.fc0(x)
            out, hidden = self.lstm(x.unsqueeze(0), hidden)                 

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x

#%%
# Hyperparameters
lr = 0.001
batch_size = 1
H = 512
L = 1024
seed = 100

##########
# INPUTS #  
##########
#(TO MODIFY)

model_path = r'C:\resnet18_LSTM_2.pt'
path_video = r"C:\video"
path_train_csv = r"C:\train.csv"
path_valid_csv = r"C:\valid.csv"
path_test_csv = r"C:\test.csv"

#%%
set_seed(seed)

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    
#%%
dico = {'Porcupine_brush_tailed': 0, 'Other': 1, 'Bongo': 2, 'Buffalo_african': 3,
        'Duiker_yellow_backed': 4, 'Duiker_blue': 5, 'Duiker_red': 6,
        'Cercopithecid': 7, 'Chevrotain_Water': 8, 'Chimpanzee': 9,
        'Civet_Genet_Nandinia': 10, 'Squirrel': 11, 'Elephant_african': 12,
        'Gorilla': 13, 'Cat_golden_Leopard_african': 14, 'Mandrill': 15, 'Mongoose': 16,
        'No_sp': 17, 'Bird': 18, 'Pangolin': 19, 'Hog_red_river': 20,
        'Rat_giant_Mouse': 21}
print(dico)

trans = [albumentations.HorizontalFlip(p=0.5),
         Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True, p=1.0),
         SmallestMaxSize(max_size=H, interpolation=1, always_apply=True, p=1.0),
         PadIfNeeded (min_height=H, min_width=L, border_mode=cv.BORDER_CONSTANT, value=[0,0,0], always_apply=True, p=1.0),
         ToTensorV2()]

trans2 = [Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True, p=1.0),
         SmallestMaxSize(max_size=H, interpolation=1, always_apply=True, p=1.0),
         PadIfNeeded (min_height=H, min_width=L, border_mode=cv.BORDER_CONSTANT, value=[0,0,0], always_apply=True, p=1.0),
         ToTensorV2()]

dataset_train = CSVVideoDataset(path_train_csv,path_video,dico,2,trans)
dataset_valid = CSVVideoDataset(path_valid_csv,path_video,dico,2,trans2)
dataset_test = CSVVideoDataset(path_test_csv,path_video,dico,2,trans2)

nbr_cl_train = len(numpy.unique(dataset_train.df['videos']))
index_train = [i for i in range(nbr_cl_train)]
nbr_cl_test = len(numpy.unique(dataset_test.df['videos']))
index_test = [i for i in range(nbr_cl_test)]
nbr_cl_valid = len(numpy.unique(dataset_valid.df['videos']))
index_valid = [i for i in range(nbr_cl_valid)]

train_sampler = SubsetRandomSampler(index_train)
test_sampler = SubsetRandomSampler(index_test)
valid_sampler = SubsetRandomSampler(index_valid)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, sampler=test_sampler)
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=1, sampler=valid_sampler)

classes = [key for key in dico]

#%%
set_seed(seed)

#model
model = CNNLSTM(nbr_cl = len(dico))
print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()
    
# specify loss function (categorical cross-entropy)
all_labels = list(dataset_train.df['labels'].values)
effectif = []

for (key,value) in dico.items():
    effectif.append(all_labels.count(value))

effectif_max = max(effectif)
weigths = [effectif_max/eff for eff in effectif]

weigths=torch.tensor(weigths)

criterion = nn.CrossEntropyLoss(weight = weigths)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

scheduler = ReduceLROnPlateau(optimizer, 'min',patience=4,min_lr = 0.000001,verbose = True)

#%%
set_seed(seed)

start = time.time()

# number of epochs to train the model
n_epochs = 50
best_epoch = 0

valid_loss_min = numpy.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):
    start2 = time.time()
    print("\n===================================================================================")
    print('Start epoch ',epoch,' : ',strftime('%H', gmtime(start2+7200)),'H',strftime('%M', gmtime(start2)),'min',strftime('%S', gmtime(start2)),'sec')
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################

    model.train()
    for video, target, name in train_loader:

        if train_on_gpu:
            target = target.cuda()
            video = video.cuda()

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        output4 = model(video)

        # calculate the batch loss
        if train_on_gpu:
            target = target.cpu()
            output4 = output4.cpu()
        loss = criterion(output4, target)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # perform a single optimization step (parameter update)
        optimizer.step()
      
        # update training loss
        train_loss += loss.item()
        
    ######################    
    # validate the model #
    ######################

    model.eval()
    with torch.no_grad():
        for video, target, name in valid_loader:

            if train_on_gpu:
                target = target.cuda()
                video = video.cuda()

            output4 = model(video)    

            # calculate the batch loss
            if train_on_gpu:
                target = target.cpu()
                output4 = output4.cpu()
            loss = criterion(output4, target)

            # update average validation loss 
            valid_loss += loss.item()
    
    # calculate average losses
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(),os.path.join(model_path),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     )
        valid_loss_min = valid_loss
        best_epoch = epoch

    print('Best epoch : ',best_epoch,' | Validation loss minimum : ', valid_loss_min)
    
    temps2 = time.time() - start2
    print("\nTemps écoulé pour l'époque",epoch,":",strftime('%H', gmtime(temps2)),'H',strftime('%M', gmtime(temps2)),'min',strftime('%S', gmtime(temps2)),'sec')

    # Scheduler
    scheduler.step(valid_loss)

#%%
temps = time.time() - start
print("Temps écoulé pour l'entrainement et la validation :",strftime('%H', gmtime(temps)),'H',strftime('%M', gmtime(temps)),'min',strftime('%S', gmtime(temps)),'sec')

#%%
set_seed(seed)

    ######################  
    # validate the model #
    ######################

model.load_state_dict(torch.load(os.path.join(model_path),map_location=torch.device('cpu')))

start4 = time.time()
print('Start validation : ',strftime('%H', gmtime(start4+7200)),'H',strftime('%M', gmtime(start4)),'min',strftime('%S', gmtime(start4)),'sec')

y_pred = []
y_true = []

# track test loss
val_loss2 = 0.0
class_correct = list(0. for i in range(len(dico)))
class_total = list(0. for i in range(len(dico)))

class_correct_top3 = list(0. for i in range(len(dico)))
class_total_top3 = list(0. for i in range(len(dico)))

model.eval()
# Initialisation of the progression bar
printProgressBar(0, len(valid_loader.dataset), prefix = 'Progress:', suffix = 'Complete', length = 50)
# iterate over test data
error = 0
with torch.no_grad():
  for ti,(data, target, name) in enumerate(valid_loader):

      if train_on_gpu:
          target = target.cuda()
          video = video.cuda()
          
      output4 = model(video)

      # calculate the batch loss
      if train_on_gpu:
          target = target.cpu()
          output4 = output4.cpu()
      loss = criterion(output4, target)

      # update test loss 
      val_loss2 += loss.item()
      # convert output probabilities to predicted class
      _, pred = torch.max(output4, 1)
      _, pred_top3 = torch.topk(output4, k = 3, dim = 1)
      # compare predictions to true label
      correct_tensor = pred.eq(target.data.view_as(pred))
      correct = numpy.squeeze(correct_tensor.numpy()) if not train_on_gpu else numpy.squeeze(correct_tensor.cpu().numpy())
      # calculate test accuracy for each object class
      label = target.data
      class_correct[label] += correct.item()
      class_total[label] += 1
      
      # Top 3
      correct_tensor_top3 = pred_top3.eq(target.data.expand_as(pred_top3))
      correct_top3 = numpy.squeeze(correct_tensor_top3.numpy()) if not train_on_gpu else numpy.squeeze(correct_tensor_top3.cpu().numpy())
      # calculate test accuracy for each object class
      label = target.data
      if True in correct_top3:
          class_correct_top3[label] += 1
      class_total_top3[label] += 1

      output5 = nn.functional.softmax(output4, dim=1)
      output6 = (output5[0]).tolist()
      output7 = [round(i,3) for i in output6]
      y_pred.append(int(pred))
      y_true.append(int(target))
      
      # Incrementation of the progression bar (in a loop)
      printProgressBar(ti + 1, len(valid_loader.dataset), prefix = 'Progress:', suffix = 'Complete', length = 50)

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
print(df_cm_val)

# average test loss
val_loss = val_loss2/len(valid_loader.dataset)
print('Validation Loss: {:.6f}\n'.format(val_loss))

vector_accuracy_val = numpy.zeros((1,len(dico)))
vector_accuracy_val_top3 = numpy.zeros((1,len(dico)))

for i in range(len(dico)):
    if class_total[i] > 0:
        print('Validation Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            numpy.sum(class_correct[i]), numpy.sum(class_total[i])))
        vector_accuracy_val[0,i] = round(100 * class_correct[i] / class_total[i],2)
    else:
        print('Validation Accuracy of %5s: N/A (no training examples)' % (classes))

print('\nValidation Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * numpy.sum(class_correct) / numpy.sum(class_total),
    numpy.sum(class_correct), numpy.sum(class_total)))

global_acc_val = 100. * numpy.sum(class_correct) / numpy.sum(class_total)

for i in range(len(dico)):
    if class_total_top3[i] > 0:
        print('Validation Accuracy top3 of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct_top3[i] / class_total_top3[i],
            numpy.sum(class_correct_top3[i]), numpy.sum(class_total_top3[i])))
        vector_accuracy_val_top3[0,i] = round(100 * class_correct_top3[i] / class_total_top3[i],2)
    else:
        print('Validation Accuracy top3 of %5s: N/A (no training examples)' % (classes))

print('\nValidation Accuracy top3 (Overall): %2d%% (%2d/%2d)' % (
    100. * numpy.sum(class_correct_top3) / numpy.sum(class_total_top3),
    numpy.sum(class_correct_top3), numpy.sum(class_total_top3)))

global_acc_val_top3 = 100. * numpy.sum(class_correct_top3) / numpy.sum(class_total_top3)

temps4 = time.time() - start4
print(" ")
print("Temps écoulé pour la validation :",strftime('%H', gmtime(temps4)),'H',strftime('%M', gmtime(temps4)),'min',strftime('%S', gmtime(temps4)),'sec')

#%%
set_seed(seed)

    ##################  
    # test the model #
    ##################

model.load_state_dict(torch.load(os.path.join(model_path),map_location=torch.device('cpu')))

start3 = time.time()
print('Start test : ',strftime('%H', gmtime(start3+7200)),'H',strftime('%M', gmtime(start3)),'min',strftime('%S', gmtime(start3)),'sec')

y_pred = []
y_true = []

# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(len(dico)))
class_total = list(0. for i in range(len(dico)))

class_correct_top3 = list(0. for i in range(len(dico)))
class_total_top3 = list(0. for i in range(len(dico)))

model.eval()
# Initialisation of the progression bar
printProgressBar(0, len(test_loader.dataset), prefix = 'Progress:', suffix = 'Complete', length = 50)
# iterate over test data
error = 0
with torch.no_grad():
  for ti,(data, target, name) in enumerate(test_loader):

      if train_on_gpu:
          target = target.cuda()
          video = video.cuda()
          
      output4 = model(video)

      # calculate the batch loss
      if train_on_gpu:
          target = target.cpu()
          output4 = output4.cpu()
      loss = criterion(output4, target)

      # update test loss 
      test_loss += loss.item()
      # convert output probabilities to predicted class
      _, pred = torch.max(output4, 1)
      _, pred_top3 = torch.topk(output4, k = 3, dim = 1)
      # compare predictions to true label
      correct_tensor = pred.eq(target.data.view_as(pred))
      correct = numpy.squeeze(correct_tensor.numpy()) if not train_on_gpu else numpy.squeeze(correct_tensor.cpu().numpy())
      # calculate test accuracy for each object class
      label = target.data
      class_correct[label] += correct.item()
      class_total[label] += 1
      
      #Top 3
      correct_tensor_top3 = pred_top3.eq(target.data.expand_as(pred_top3))
      correct_top3 = numpy.squeeze(correct_tensor_top3.numpy()) if not train_on_gpu else numpy.squeeze(correct_tensor_top3.cpu().numpy())
      # calculate test accuracy for each object class
      label = target.data
      if True in correct_top3:
          class_correct_top3[label] += 1
      class_total_top3[label] += 1

      output5 = nn.functional.softmax(output4, dim=1)
      output6 = (output5[0]).tolist()
      output7 = [round(i,3) for i in output6]
      y_pred.append(int(pred))
      y_true.append(int(target))
      
      # Incrementation of the progression bar (in a loop)
      printProgressBar(ti + 1, len(test_loader.dataset), prefix = 'Progress:', suffix = 'Complete', length = 50)

mat_conf = confusion_matrix(y_true = y_true,y_pred = y_pred)
conf = numpy.zeros((len(dico)+1,len(dico)+2),dtype=object)
conf[len(conf)-1,0] = 'total'
for i in range(len(dico)):
    conf[i,0] = classes[i]
    for j in range(len(dico)):
            conf[i,j+1] = mat_conf[i,j]
            conf[i,len(dico)+1] += mat_conf[i,j]
            conf[len(dico),len(dico)+1] += mat_conf[i,j]
            conf[len(dico),j+1] += mat_conf[i,j]

df_cm = pd.DataFrame(conf,columns=columns)
print(df_cm)

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

vector_accuracy = numpy.zeros((1,len(dico)))
vector_accuracy_top3 = numpy.zeros((1,len(dico)))

for i in range(len(dico)):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            numpy.sum(class_correct[i]), numpy.sum(class_total[i])))
        vector_accuracy[0,i] = round(100 * class_correct[i] / class_total[i],2)
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * numpy.sum(class_correct) / numpy.sum(class_total),
    numpy.sum(class_correct), numpy.sum(class_total)))
print(" ")

for i in range(len(dico)):
    if class_total_top3[i] > 0:
        print('Test Accuracy top3 of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct_top3[i] / class_total_top3[i],
            numpy.sum(class_correct_top3[i]), numpy.sum(class_total_top3[i])))
        vector_accuracy_top3[0,i] = round(100 * class_correct_top3[i] / class_total_top3[i],2)
    else:
        print('Test Accuracy top3 of %5s: N/A (no training examples)' % (classes))

print('\nTest Accuracy top3 (Overall): %2d%% (%2d/%2d)' % (
    100. * numpy.sum(class_correct_top3) / numpy.sum(class_total_top3),
    numpy.sum(class_correct_top3), numpy.sum(class_total_top3)))

temps3 = time.time() - start3
print(" ")
print("Temps écoulé pour le test :",strftime('%H', gmtime(temps3)),'H',strftime('%M', gmtime(temps3)),'min',strftime('%S', gmtime(temps3)),'sec')
