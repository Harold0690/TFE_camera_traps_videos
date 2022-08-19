# TFE_camera_traps_videos

The goal of this study is to automate species recognition in camera traps videos. Those camera traps were installed in tropical forests in Central Africa.

The videos are in AVI or MP4 format, last 5 or 30 seconds, are composed of 30 frames per second and have a size of 1920 x 1080 pixels. These videos were captured by the Bolyguard SG 2060X model (Boly, Victoriaville, QC, Canada).

Binary classification (animal / background) and multi-species classification (22 classes) were tested. 

The 22 classes are listed in the photo below.
![image](https://user-images.githubusercontent.com/101332012/185503411-0e26c23a-28f9-417f-9235-cfb428743a05.png)

I also tested 8 model architectures :
    - Architecture 1 : ResNet-18 + ConvLSTM + MLP (multi-species classification);
    - Architecture 2 : ResNet-18 + LSTM + MLP (multi-species classification);
    - Architecture 3 : ResNet-18 + MLP (multi-species classification);
    - Architecture 4 : ResNet-18 + MLP (binary classification);
    - Architecture 5 : ResNet-18 + MLP (multi-species classification);
    - Architecture 6 : Architecture 4 + Architecture 5 (multi-species classification);
    - Architecture 7 : ResNet3D-50 (multi-species classification);
    - Architecture 8 : ResNet3D-50 (binary classification).
    
The architectures 1 and 2 were a failure. 
The architectures 3, 4, 5 and 6 are image-based classifiers and reach the best results.
The architectures 7 and 8 show promising results but need furhter research.

Architecture 3 performed best for multi-species classification. The model trained with this architecture was named ResNet2D 1. Its performance by class is shown in the following table.
![image](https://user-images.githubusercontent.com/101332012/185509937-bc2fc253-91fd-45d1-b79f-238e494f42da.png)


For binary classification, architecture 4 was chosen as it achieved the best results. The model trained with this architecture was named ResNet2D 2 and its performance per class is shown in the following table.
![image](https://user-images.githubusercontent.com/101332012/185509880-808f5412-5ce8-4845-999c-6d2a2e3333be.png)

The conversion of image-level predictions into a single video-level prediction is achieved using an empirical method.

You can find in this repertory, the scripts to train models with the 8 architectures and the scripts to use the 2 selected models.

The inputs for the training scripts are the path to the directory containing the videos, the path to the model to be saved and the path to the CSV file of each dataset. This CSV file must contain a column named 'videos' with the names of the videos and a column named 'labels' with the number of the class corresponding to each video.

The inputs for the scripts that allow the use of the templates are the path to the directory containing the videos and the path to the model. The output of those scripts is a CSV file with for each analyzed video: the name of the video, the top1 prediction and the top3 prediction.

I couldn't share ResNet2D 1 and ResNet2D 2 in this repertory because the files are too big. But if you want it or if you want more information about my work, you can contact me via the following email address : harold.campers@gmail.com
