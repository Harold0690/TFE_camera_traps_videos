#%%
import torch
import os
import numpy
import albumentations
import cv2 as cv
import pandas as pd  # variable non politiquement correct !!! Sam.
import random
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from time import strftime,gmtime
import torch.nn as nn
from albumentations import SmallestMaxSize,PadIfNeeded,Normalize
import wandb
from sklearn.metrics import confusion_matrix
import shutil

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
class ImageCSVFolder(Dataset):

    def __init__(self, img_dir, csv_file, transform=None):
        self.csv_file = (pd.read_csv(csv_file,delimiter=';',index_col=False,header=0)).values
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.csv_file[idx, 0])
        image = Image.open(img_path)
        image = numpy.array(image)

        label = self.csv_file[idx, 1]

        name = self.csv_file[idx, 0]

        if self.transform:
            transform_pipeline = albumentations.Compose(self.transform)
            image = transform_pipeline(image=image)["image"]

        return image, label, name

#%%
wandb_lr = 0.001
wandb_batch_size = 2
wandb_H = 1080
wandb_L = 1920
wandb_model_name = 'resnet18_3_1.pt'
wandb_seed = 100

#%%
set_seed(wandb_seed)

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

#%%
dico = {'Athérure': 0, 'Autre': 1, 'Bongo': 2, 'Buffle de forêt': 3,
        'Céphalophe à dos jaune': 4, 'Céphalophe bleu': 5, 'Céphalophe rouge': 6,
        'Cercopithecidae': 7, 'Chevrotain aquatique': 8, 'Chimpanzé': 9,
        'Civette/Genette/Nandinie': 10, 'Écureuil': 11, 'Eléphant de forêt': 12,
        'Gorille': 13, 'Grand félin': 14, 'Mandrill': 15, 'Mangouste': 16,
        'Oiseau': 17, 'Pangolin': 18, 'Potamochère': 19, 'Rat géant/Souris': 20}
print(dico)

path_in = r'D:\Harold\model_video\frame20'
path_csv_train = r'D:\Harold\model_video\img_train20_3.csv'
path_csv_test = r'D:\Harold\model_video\img_test20_3.csv'
path_csv_valid = r'D:\Harold\model_video\img_valid20_3.csv'

mean =[0.486, 0,509, 0.510]
std =[0.198, 0.203, 0.203]

trans = [albumentations.HorizontalFlip(p=0.5),
          Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True, p=1.0),
          SmallestMaxSize(max_size=wandb_H, interpolation=1, always_apply=True, p=1.0),
          PadIfNeeded (min_height=wandb_H, min_width=wandb_L, border_mode=cv.BORDER_CONSTANT, value=[0,0,0], always_apply=True, p=1.0),
          ToTensorV2()]

# trans = [albumentations.HorizontalFlip(p=0.5),
#          albumentations.augmentations.transforms.ColorJitter (brightness=0.5, contrast=0.5, saturation=0.0, hue=0.0, always_apply=False, p=0.25),
#          albumentations.augmentations.transforms.Blur (blur_limit=11, always_apply=False, p=0.25),
#          Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True, p=1.0),
#          SmallestMaxSize(max_size=512, interpolation=1, always_apply=True, p=1.0),
#          PadIfNeeded (min_height=512, min_width=1024, border_mode=cv.BORDER_CONSTANT, value=[0,0,0], always_apply=True, p=1.0),
#          ToTensorV2()]

trans2 = [Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True, p=1.0),
          SmallestMaxSize(max_size=wandb_H, interpolation=1, always_apply=True, p=1),
          PadIfNeeded (min_height=wandb_H, min_width=wandb_L, border_mode=cv.BORDER_CONSTANT, value=[0,0,0], always_apply=True, p=1.0),
          ToTensorV2()]

data_train = ImageCSVFolder(img_dir=path_in,csv_file=path_csv_train,transform=trans)
data_test = ImageCSVFolder(img_dir=path_in,csv_file=path_csv_test,transform=trans2)
data_valid = ImageCSVFolder(img_dir=path_in,csv_file=path_csv_valid,transform=trans2)

nbr_cl_train = len(data_train)
index_train = [i for i in range(nbr_cl_train)]
nbr_cl_test = len(data_test)
index_test = [i for i in range(nbr_cl_test)]
nbr_cl_valid = len(data_valid)
index_valid = [i for i in range(nbr_cl_valid)]

train_sampler = SubsetRandomSampler(index_train)
test_sampler = SubsetRandomSampler(index_test)
valid_sampler = SubsetRandomSampler(index_valid)

train_loader = torch.utils.data.DataLoader(data_train, batch_size=wandb_batch_size, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=1, sampler=test_sampler)
valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=1, sampler=valid_sampler)

classes = [key for key in dico]

#%%
set_seed(wandb_seed)

# create a complete CNN
model = models.resnet18(pretrained=True)
num_backbone_features = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(p=0.5),
                         nn.Linear(num_backbone_features, 256, bias=True),
                         nn.Dropout(p=0.5),
                         nn.Linear(256, len(dico), bias=True))
print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()

# specify loss function (categorical cross-entropy)
all_labels = list(data_train.csv_file[:,1])
effectif = []

for (key,value) in dico.items():
    effectif.append(all_labels.count(value))

effectif_max = max(effectif)
weigths = [effectif_max/eff for eff in effectif]

weigths=torch.tensor(weigths)
  
criterion = nn.CrossEntropyLoss(weight = weigths)

# specify optimizer
optimizer = optim.SGD(model.parameters(), lr = wandb_lr, momentum=0.9)#

scheduler = ReduceLROnPlateau(optimizer, 'max',patience=4,min_lr = 0.000001,verbose = True)
    
#%%
set_seed(wandb_seed)

start = time.time()

# number of epochs to train the model
n_epochs = 50
best_epoch = 0

valid_loss_min = numpy.Inf # track change in validation loss
valid_acc_max = 0

#WANDB
wandb.init(project="TFE", entity="harold06")

wandb.config.PC = 'PC59'
wandb.config.seed = wandb_seed
wandb.config.lr = wandb_lr
wandb.config.nbr_classes = len(dico)
wandb.config.optimizer = "SGD"
wandb.config.scheduler = "ReduceLROnPlateau"
wandb.config.loss_fucntion = "CrossEntropyLoss"
wandb.config.batch_size = wandb_batch_size
wandb.config.model = "Resnet18"
wandb.config.backbone = '512 => dropout(0.5) => 256 => dropout(0.5) => ' + str(len(dico))
wandb.config.size = str(wandb_H)+"X"+str(wandb_L)
wandb.config.transformation = str(trans)
wandb.config.model_name = wandb_model_name
wandb.config.size_dataset = len(train_loader.sampler)+len(valid_loader.sampler)+len(test_loader.sampler)

for epoch in range(1, n_epochs+1):
    start2 = time.time()
    print("\n===================================================================================")
    print('Start epoch ',epoch,' : ',strftime('%H', gmtime(start2+7200)),'H',strftime('%M', gmtime(start2)),'min',strftime('%S', gmtime(start2)),'sec')
    # keep track of training and validation loss
    train_loss = 0.0
    train_loss2 = 0.0
    valid_loss = 0.0
    valid_loss2 = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, target,name in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        # print(type(data))
        output = model(data)
        # print(nn.functional.softmax(output, dim=1),target)
        # calculate the batch loss
        # print(target)
        if train_on_gpu:
            target = target.cpu()
            output = output.cpu()
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # for param in model.parameters():
            # print((param[0][0]))
            # break
        # update training loss
        train_loss += loss.item()*data.size(0)
        # print(loss)
        # print(loss.item())
        # print(data.size(0))
    #     train_loss2 += loss.item()
    # print(len(train_loader.sampler))
    # print( train_loss2)
    # print( train_loss)
    # print(train_loss2/len(data_train.csv_file))
    # print(train_loss/len(train_loader.sampler))
        
    ######################    
    # validate the model #
    ######################
    
    class_correct = list(0. for i in range(len(dico)))
    class_total = list(0. for i in range(len(dico)))
    
    model.eval()
    with torch.no_grad():
      for data, target, name in valid_loader:
          # move tensors to GPU if CUDA is available
          if train_on_gpu:
              data, target = data.cuda(), target.cuda()
          # forward pass: compute predicted outputs by passing inputs to the model
          output = model(data)
          # calculate the batch loss
          if train_on_gpu:
              target = target.cpu()
              output = output.cpu()
          loss = criterion(output, target)
          _, pred = torch.max(output, 1)  
          # update average validation loss 
          valid_loss += loss.item()*data.size(0)
          _, pred = torch.max(output, 1)
          _, pred_top3 = torch.topk(output, k = 3, dim = 1)
          # compare predictions to true label
          correct_tensor = pred.eq(target.data.view_as(pred))
          correct = numpy.squeeze(correct_tensor.numpy()) if not train_on_gpu else numpy.squeeze(correct_tensor.cpu().numpy())
          # calculate test accuracy for each object class
          label = target.data
          class_correct[label] += correct.item()
          class_total[label] += 1
          # print(loss)
          # print(loss.item())
          # print(data.size(0))
    #       valid_loss2 += loss.item()
    # print(len(valid_loader.sampler))
    # print( valid_loss2)
    # print( valid_loss)
    # print( valid_loss2/len(data_valid.csv_file))
    # print( valid_loss/len(valid_loader.sampler))
    
    valid_accuracy = numpy.sum(class_correct) / numpy.sum(class_total)
    
    # calculate average losses
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
    
    wandb.log({"Training_loss": train_loss,"Validation_loss": valid_loss, "Validation accuracy":valid_accuracy})
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Accuracy: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_accuracy, valid_loss))
    
    # save model if validation accuracy has increased
    if valid_accuracy >= valid_acc_max:
        print('Validation accuracy increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_acc_max,valid_accuracy))
        torch.save(model.state_dict(), os.path.join(r'D:\Harold\model_video\resnet_model',wandb_model_name))
        valid_acc_max = valid_accuracy
        best_epoch = epoch
    
    print('Best epoch : ',best_epoch,' | Validation accuracy maximum : ', valid_acc_max)
    
    # # save model if validation loss has decreased
    # if valid_loss <= valid_loss_min:
    #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
    #     valid_loss_min,
    #     valid_loss))
    #     torch.save(model.state_dict(), os.path.join(r'D:\Harold\model_video\resnet_model',wandb_model_name))
    #     valid_loss_min = valid_loss
    #     best_epoch = epoch
    
    # print('Best epoch : ',best_epoch,' | Validation loss minimum : ', valid_loss_min)
    
    temps2 = time.time() - start2
    print("Temps écoulé pour l'époque",epoch,":",strftime('%H', gmtime(temps2)),'H',strftime('%M', gmtime(temps2)),'min',strftime('%S', gmtime(temps2)),'sec')

    # Scheduler
    scheduler.step(valid_accuracy)

#%%
temps = time.time() - start
print("Temps écoulé pour l'entrainement et la validation :",strftime('%H', gmtime(temps)),'H',strftime('%M', gmtime(temps)),'min',strftime('%S', gmtime(temps)),'sec')

#%%
set_seed(wandb_seed)

model.load_state_dict(torch.load(os.path.join(r'D:\Harold\model_video\resnet_model',wandb_model_name),map_location=torch.device('cpu')))

start4 = time.time()
print('Start validation : ',strftime('%H', gmtime(start4+7200)),'H',strftime('%M', gmtime(start4)),'min',strftime('%S', gmtime(start4)),'sec')

path_error = r'D:\Harold\model_video\error_validation'
if os.path.exists(path_error):
    shutil.rmtree(path_error)
    os.makedirs(path_error)
else:
    os.makedirs(path_error)

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
with torch.no_grad():
    for ti,(data, target, name) in enumerate(valid_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        if train_on_gpu:
            target = target.cpu()
            output = output.cpu()
        loss = criterion(output, target)
        # update test loss 
        val_loss2 += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        # output = nn.functional.softmax(output, dim=1)
        _, pred = torch.max(output, 1)
        _, pred_top3 = torch.topk(output, k = 3, dim = 1)
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
        
        output5 = nn.functional.softmax(output, dim=1)
        output6 = (output5[0]).tolist()
        output7 = [round(i,3) for i in output6]
        # print("\n===================================================================================")
        # print('Output : ',output7)
        # print("Target : ",classes[target]," | Prediction : ",classes[pred]," | Probability : ",round(((output5[0]).tolist())[pred]*100,3)," %")      
        # print("===================================================================================")
        y_pred.append(int(pred))
        y_true.append(int(target))
        
        if int(target) != int(pred):
            target10 = str(classes[target])
            if target10 == "Civette/Genette/Nandinie":
                target10 = "Civette-Genette-Nandinie"
            elif target10 == "Rat géant/Souris":
                target10 = "Rat géant-Souris"
            pred10 = str(classes[pred])
            if pred10 == "Civette/Genette/Nandinie":
                pred10 = "Civette-Genette-Nandinie"
            elif pred10 == "Rat géant/Souris":
                pred10 = "Rat géant-Souris"
            shutil.copy(os.path.join(path_in,str(name[0])),os.path.join(path_error,target10+"_"+pred10+"_"+str(name[0])))

        
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
print('validation Loss: {:.6f}\n'.format(val_loss))

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
print(" ")

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
set_seed(wandb_seed)

model.load_state_dict(torch.load(os.path.join(r'D:\Harold\model_video\resnet_model',wandb_model_name),map_location=torch.device('cpu')))

start3 = time.time()
print('Start test : ',strftime('%H', gmtime(start3+7200)),'H',strftime('%M', gmtime(start3)),'min',strftime('%S', gmtime(start3)),'sec')

path_error = r'D:\Harold\model_video\error_test'
if os.path.exists(path_error):
    shutil.rmtree(path_error)
    os.makedirs(path_error)
else:
    os.makedirs(path_error)

y_pred = []
y_true = []

# track test loss
test_loss2 = 0.0
class_correct = list(0. for i in range(len(dico)))
class_total = list(0. for i in range(len(dico)))

class_correct_top3 = list(0. for i in range(len(dico)))
class_total_top3 = list(0. for i in range(len(dico)))

model.eval()
# Initialisation of the progression bar
printProgressBar(0, len(test_loader.dataset), prefix = 'Progress:', suffix = 'Complete', length = 50)
# iterate over test data
with torch.no_grad():
    for ti,(data, target, name) in enumerate(test_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        if train_on_gpu:
            target = target.cpu()
            output = output.cpu()
        loss = criterion(output, target)
        # update test loss 
        test_loss2 += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        # output = nn.functional.softmax(output, dim=1)
        _, pred = torch.max(output, 1)
        _, pred_top3 = torch.topk(output, k = 3, dim = 1)
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
        
        output5 = nn.functional.softmax(output, dim=1)
        output6 = (output5[0]).tolist()
        output7 = [round(i,3) for i in output6]
        # print("\n===================================================================================")
        # print('Output : ',output7)
        # print("Target : ",classes[target]," | Prediction : ",classes[pred]," | Probability : ",round(((output5[0]).tolist())[pred]*100,3)," %")      
        # print("===================================================================================")
        y_pred.append(int(pred))
        y_true.append(int(target))
        
        # if int(target) != int(pred):
            # target10 = str(classes[target])
            # if target10 == "Civette/Genette/Nandinie":
            #     target10 = "Civette-Genette-Nandinie"
            # elif target10 == "Rat géant/Souris":
            #     target10 = "Rat géant-Souris"
            # pred10 = str(classes[pred])
            # if pred10 == "Civette/Genette/Nandinie":
            #     pred10 = "Civette-Genette-Nandinie"
            # elif pred10 == "Rat géant/Souris":
            #     pred10 = "Rat géant-Souris"
        #     shutil.copy(os.path.join(path_in,str(name[0])),os.path.join(path_error,target10+"_"+pred10+"_"+str(name[0])))

        
        # Incrementation of the progression bar (in a loop)
        printProgressBar(ti + 1, len(test_loader.dataset), prefix = 'Progress:', suffix = 'Complete', length = 50)
     
        
mat_conf = confusion_matrix(y_true = y_true,y_pred = y_pred)
conf = numpy.zeros((len(dico)+1,len(dico)+2),dtype=object)
conf[len(conf)-1,0] = 'total'
for i in range(len(dico)):
    conf[i,0] = classes[i]
    for j in range(len(dico)):
            conf[i,j+1] = mat_conf[i,j]
            # print(mat_conf[i,j])
            conf[i,len(dico)+1] += mat_conf[i,j]
            conf[len(dico),len(dico)+1] += mat_conf[i,j]
            conf[len(dico),j+1] += mat_conf[i,j]

df_cm = pd.DataFrame(conf,columns=columns)
print(df_cm)

# average test loss
test_loss = test_loss2/len(test_loader.dataset)
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

#%%
# wandb.summary['Correction_data_augmentation'] = str(trans)
wandb.summary['Time_training+validation'] = (strftime('%H', gmtime(temps))+' : '+strftime('%M', gmtime(temps))+' : '+strftime('%S', gmtime(temps)))
wandb.summary['Average_Time_training+validation'] = (strftime('%H', gmtime(temps2))+' : '+strftime('%M', gmtime(temps2))+' : '+strftime('%S', gmtime(temps2)))
wandb.summary['Epoch'] = epoch-1
wandb.summary['Best_epoch'] = best_epoch
wandb.summary['Time_validation'] = (strftime('%H', gmtime(temps4))+' : '+strftime('%M', gmtime(temps4))+' : '+strftime('%S', gmtime(temps4)))
wandb.summary['Validation_loss'] = val_loss
wandb.summary['Global_accuracy_validation'] = global_acc_val
wandb.summary['Vector_accuracy_validation'] = wandb.Table(data = vector_accuracy_val, columns=classes)
wandb.summary['Confusion_matrix_validation'] = wandb.Table(columns=columns, data = conf_val)
wandb.summary['Global_accuracy_validation_top3'] = global_acc_val_top3
wandb.summary['Vector_accuracy_validation_top3'] = wandb.Table(data = vector_accuracy_val_top3, columns=classes)
wandb.summary['Time_test'] = (strftime('%H', gmtime(temps3))+' : '+strftime('%M', gmtime(temps3))+' : '+strftime('%S', gmtime(temps3)))
wandb.summary['Test_loss'] = test_loss
wandb.summary['Global_accuracy_test'] = round(100. * numpy.sum(class_correct) / numpy.sum(class_total),1)
wandb.summary['Vector_accuracy_test'] = wandb.Table(data = vector_accuracy, columns=classes)
wandb.summary['Confusion_matrix_test'] = wandb.Table(columns=columns, data = conf)
wandb.summary['Global_accuracy_test_top3'] = round(100. * numpy.sum(class_correct_top3) / numpy.sum(class_total_top3),1)
wandb.summary['Vector_accuracy_test_top3'] = wandb.Table(data = vector_accuracy_top3, columns=classes)
wandb.finish()
