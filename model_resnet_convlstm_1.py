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
import wandb
from sklearn.metrics import confusion_matrix
# import shutil

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
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

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

        self._img_names = numpy.unique(self.df['images'])
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
        annotations = self.df[self.df['images'] == img_name]
        annotations = annotations.drop(columns='images')

        annotations = list(annotations['labels'])
        target = self.dico_labels.get(annotations[0])

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
class CameraTrapMultiLabels(nn.Module):
    ''' Classification of multi-labels videos '''

    def __init__(
        self,
        path_resnet_model: str,
        nbr_cl: int,
        pretrained: bool = True, 
        batch_size: int = 1,
        kernel_size: tuple = 3, 
        channel: int = 512,
        gpu: bool = True
        ):
        '''
        Args:
            nbr_cl (int): number of output classes, background included. 
                Defaults to 2.
            pretrained (bool, optional): set False to disable pretrained DLA encoder parameters
                from ImageNet. Defaults to True.
            batch_size (int): size of the batch.
                Defaults to 1.
            kernel_size (tuple): size of the kernel for the convolution.
                Defaults to (3,3).
            channel (int): number of channels of the image that comes out of the encoder.
                Defaults to 512.
            gpu (bool): 'True' means that gpu is available.
                Defaults to True.
        '''

        super(CameraTrapMultiLabels, self).__init__()

        self.batch_size = batch_size
        self.channel = channel
        self.kernel_size = kernel_size
        self.nbr_cl = nbr_cl
        self.pretrained = pretrained
        self.gpu = gpu

        #Encoder
        model_detection = models.resnet18(pretrained=True)
        num_backbone_features = model_detection.fc.in_features
        model_detection.fc = nn.Sequential(nn.Linear(num_backbone_features, nbr_cl, bias=True))

        model_detection.load_state_dict(torch.load(path_resnet_model,map_location=torch.device('cpu')))
        model_detection_dict = model_detection.state_dict()

        encoder = models.resnet18(pretrained=self.pretrained)
        encoder = list(encoder.children())
        self.encoder = nn.Sequential(*encoder[:-2])

        encoder_dict = self.encoder.state_dict()

        filtered_dict = {k: v for k, v in model_detection_dict.items() if k in encoder_dict}
        encoder_dict.update(filtered_dict)
        self.encoder.load_state_dict(encoder_dict)

        #ConvLSTM
        self.convlstm = ConvLSTM(input_dim = self.channel, hidden_dim = self.channel, kernel_size = self.kernel_size, num_layers = 3, batch_first=True, bias=True, return_all_layers=False)

        #Fully connected layers
        self.fc = nn.Sequential(nn.BatchNorm2d(self.channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                           nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                           nn.Flatten(),
                           nn.Dropout(p=0.25),
                           nn.Linear(in_features=self.channel, out_features=256, bias=True),
                           nn.ReLU(inplace=True),
                           nn.Linear(in_features=256, out_features=128, bias=True),
                           nn.ReLU(inplace=True),
                           nn.Linear(in_features=128, out_features=64, bias=True),
                           nn.ReLU(inplace=True),
                           nn.Linear(in_features=64, out_features=self.nbr_cl, bias=True))
        
    def forward(self, input: torch.Tensor):
        for t in range(len(input[0])):
          with torch.no_grad():
              output1 = (self.encoder(input[:,t,:,:,:]))
          if t == 0:
            output11 = output1
          else:
            output11 = torch.cat((output11,output1),0)
        # print(output11[0][0])
        output11 = torch.unsqueeze(output11,0)
        #print(output11.shape)


        # for frame in loader:
        #     if self.gpu:
        #         frame = frame.cuda()
        #     output1 = self.encoder(frame)
        _, last_states = self.convlstm(output11)
            # cur_state = output2
        #print(len(last_states[0][0]))
        # print(last_states[0][0].shape)#[len(input[0])-1]
        #print(last_states[0][0])
        #output3 = torch.unsqueeze(output2[0][len(output2[0])-1],0)
        # print(last_states[0][0][0][0])
        output4 = self.fc(last_states[0][0])

        return output4

#%%
wandb_lr = 0.0001
wandb_batch_size = 1
wandb_H = 512
wandb_L = 1024
wandb_resnet_model_name = 'resnet18_4.pt'
wandb_model_name = 'resnet18_ConvLSTM_1.pt'
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
dico = {'Athérure': 0, 'Autres mangoustes': 1, 'Bongo': 2,
        'Buffle de forêt': 3, 'Cercopithecidae': 4,
        'Chevrotain aquatique': 5, 'Chimpanzé': 6,
        'Céphalophe bleu': 7, 'Céphalophe rouge': 8,
        'Céphalophe à dos jaune': 9, 'Eléphant de forêt': 10,
        'Genette': 11, 'Gorille': 12, 'No_sp': 13, 'Oiseau': 14,
        'Potamochère': 15,'Rat géant': 16, 'Écureuil': 17}
print(dico)

trans = [albumentations.HorizontalFlip(p=0.5),
         Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True, p=1.0),
         SmallestMaxSize(max_size=wandb_H, interpolation=1, always_apply=True, p=1.0),
         PadIfNeeded (min_height=wandb_H, min_width=wandb_L, border_mode=cv.BORDER_CONSTANT, value=[0,0,0], always_apply=True, p=1.0),
         ToTensorV2()]
trans2 = [Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True, p=1.0),
         SmallestMaxSize(max_size=wandb_H, interpolation=1, always_apply=True, p=1.0),
         PadIfNeeded (min_height=wandb_H, min_width=wandb_L, border_mode=cv.BORDER_CONSTANT, value=[0,0,0], always_apply=True, p=1.0),
         ToTensorV2()]

path_video = r"D:\Harold\model_video\video_model2"
path_train_csv = r"D:\Harold\model_video\train3.csv"
path_valid_csv = r"D:\Harold\model_video\valid3.csv"
path_test_csv = r"D:\Harold\model_video\test3.csv"

dataset_train = CSVVideoDataset(path_train_csv,path_video,dico,2,trans)
dataset_valid = CSVVideoDataset(path_valid_csv,path_video,dico,2,trans2)
dataset_test = CSVVideoDataset(path_test_csv,path_video,dico,2,trans2)

nbr_cl_train = len(numpy.unique(dataset_train.df['images']))
index_train = [i for i in range(nbr_cl_train)]
nbr_cl_test = len(numpy.unique(dataset_test.df['images']))
index_test = [i for i in range(nbr_cl_test)]
nbr_cl_valid = len(numpy.unique(dataset_valid.df['images']))
index_valid = [i for i in range(nbr_cl_valid)]

train_sampler = SubsetRandomSampler(index_train)
test_sampler = SubsetRandomSampler(index_test)
valid_sampler = SubsetRandomSampler(index_valid)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=wandb_batch_size, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, sampler=test_sampler)
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=1, sampler=valid_sampler)

classes = [key for key in dico]

#%%
set_seed(wandb_seed)

#model
path_resnet_model = os.path.join(r"D:\Harold\model_video\resnet_model",wandb_resnet_model_name)
model = CameraTrapMultiLabels(path_resnet_model=path_resnet_model, nbr_cl = len(dico), pretrained = True, batch_size = 1, kernel_size = (3,3), gpu = train_on_gpu)
print(model)

if train_on_gpu:
    model.cuda()
  
# specify loss function (categorical cross-entropy)
all_labels = list(dataset_train.df['labels'].values)
all_labels2 = [dico.get(i) for i in all_labels]
effectif = []

for (key,value) in dico.items():
    effectif.append(all_labels2.count(value))

effectif_max = max(effectif)
weigths = [effectif_max/eff for eff in effectif]

weigths=torch.tensor(weigths)
  
criterion = nn.CrossEntropyLoss(weight = weigths)

optimizer = optim.SGD(model.parameters(), lr=wandb_lr, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = ReduceLROnPlateau(optimizer, 'min',patience=4,min_lr = 0.000001,verbose = True)

#%%
set_seed(wandb_seed)

start = time.time()

# number of epochs to train the model
n_epochs = 50
best_epoch = 0

valid_loss_min = numpy.Inf # track change in validation loss

#WANDB
wandb.init(project="TFE", entity="harold06")

wandb.config.sens_video = 'no reverse'
wandb.config.PC = 'PC59'
wandb.config.seed = wandb_seed
wandb.config.lr = wandb_lr
wandb.config.nbr_classes = len(dico)
wandb.config.optimizer = "SGD"
wandb.config.scheduler = "ReduceLROnPlateau"
wandb.config.loss_fucntion = "CrossEntropyLoss"
wandb.config.batch_size = wandb_batch_size
wandb.config.model = "Resnet18 + LSTM"
wandb.config.size = str(wandb_H)+"X"+str(wandb_L)
wandb.config.transformation = str(trans)
wandb.config.resnet_model_name = wandb_resnet_model_name
wandb.config.model_name = wandb_model_name
wandb.config.size_dataset = len(train_loader.sampler)+len(valid_loader.sampler)+len(test_loader.sampler)

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
        # target = target[0]
        # input = video[0]
        # print(len(input))

        if train_on_gpu:
            target = target.cuda()
            video = video.cuda()

        # print(video[0][0][0][0])
        # print(video.shape)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        output4 = model(video)
        # print(nn.functional.softmax(output4, dim=1),target)
        # print(output4)
        
        #output4 = torch.unsqueeze(output4,0)
        # print(output4,target)
        # calculate the batch loss
        
        if train_on_gpu:
            output4 = output4.cpu()
            target = target.cpu()
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
            # target = target[0]
            # input = video[0]

            if train_on_gpu:
                target = target.cuda()
                video = video.cuda()
            
            output4 = model(video)
            # print(nn.functional.softmax(output4, dim=1),target)
            # output4 = torch.unsqueeze(output4,0)
            # print(output4,target)

            # calculate the batch loss
            if train_on_gpu:
                output4 = output4.cpu()
                target = target.cpu()
            loss = criterion(output4, target)

            # update average validation loss 
            valid_loss += loss.item()
    
    # calculate average losses
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
    
    wandb.log({"Training_loss": train_loss,"Validation_loss": valid_loss})
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(),os.path.join(r'D:\Harold\model_video\model', wandb_model_name) ,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     )
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
set_seed(wandb_seed)

    ######################  
    # validate the model #
    ######################

model.load_state_dict(torch.load(os.path.join(r'D:\Harold\model_video\model', wandb_model_name),map_location=torch.device('cpu')))

start4 = time.time()
print('Start validation : ',strftime('%H', gmtime(start4+7200)),'H',strftime('%M', gmtime(start4)),'min',strftime('%S', gmtime(start4)),'sec')

# path_error = r'D:\Harold\model_video\resnet_lstm_error_validation'
# if os.path.exists(path_error):
#     shutil.rmtree(path_error)
#     os.makedirs(path_error)
# else:
#     os.makedirs(path_error)

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
      # target = target[0]
      # input = video[0]

      if train_on_gpu:
          target = target.cuda()
          video = video.cuda()
          
      output4 = model(video)

      # output4 = torch.unsqueeze(output4,0)
      # calculate the batch loss
      #print(output4,target)
      if train_on_gpu:
          output4 = output4.cpu()
          target = target.cpu()
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
      # print("\n===================================================================================")
      # print('Output : ',output7)
      # print("Target : ",dataset_train.labels[target]," | Prediction : ",dataset_train.labels[pred]," | Probability : ",round(((output5[0]).tolist())[pred]*100,3)," %")      
      # print("===================================================================================")
      y_pred.append(int(pred))
      y_true.append(int(target))
      
      # if int(target) != int(pred):
      #     shutil.copy(os.path.join(path_in,str(name[0])),os.path.join(path_error,str(classes[target])+"_"+str(classes[pred])+"_"+str(name[0])))

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

    ##################  
    # test the model #
    ##################

model.load_state_dict(torch.load(os.path.join(r'D:\Harold\model_video\model', wandb_model_name),map_location=torch.device('cpu')))

start3 = time.time()
print('Start test : ',strftime('%H', gmtime(start3+7200)),'H',strftime('%M', gmtime(start3)),'min',strftime('%S', gmtime(start3)),'sec')

# path_error = r'D:\Harold\model_video\model\resnet_lstm_error_test'
# if os.path.exists(path_error):
#     shutil.rmtree(path_error)
#     os.makedirs(path_error)
# else:
#     os.makedirs(path_error)

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
      # target = target[0]
      # input = video[0]

      if train_on_gpu:
          target = target.cuda()
          video = video.cuda()
          
      output4 = model(video)

      # output4 = torch.unsqueeze(output4,0)
      # calculate the batch loss
      #print(output4,target)
      if train_on_gpu:
          output4 = output4.cpu()
          target = target.cpu()
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
      # print("\n===================================================================================")
      # print('Output : ',output7)
      # print("Target : ",dataset_train.labels[target]," | Prediction : ",dataset_train.labels[pred]," | Probability : ",round(((output5[0]).tolist())[pred]*100,3)," %")      
      # print("===================================================================================")
      y_pred.append(int(pred))
      y_true.append(int(target))
      
      # if int(target) != int(pred):
      #     shutil.copy(os.path.join(path_in,str(name[0])),os.path.join(path_error,str(classes[target])+"_"+str(classes[pred])+"_"+str(name[0])))

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
