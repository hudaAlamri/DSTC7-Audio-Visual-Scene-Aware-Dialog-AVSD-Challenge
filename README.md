Dialog System Technology Challenges 7 (DSTC7) Track 3
# Audio Visual Scene-Aware Dialog (AVSD)
Track description paper: Pease cite this paper if you will use the shared data sets.

https://arxiv.org/abs/1806.00525  

## News:

### - Registration 
   Please register:  https://docs.google.com/forms/d/e/1FAIpQLSf4aoCdtLsnFr_AKfp3tnTy4OUCITy5avcEEpUHJ9oZ5ZFvbg/viewform  
   Please let us share the data with you using your registered e-mail.

### - Data release
   Prototype datasets: 6172(training), 732(validation), 733(test)
   https://drive.google.com/drive/u/2/folders/1JGE4eeelA0QBA7BwYvj89kSClE3f9k65
   
         - QA text dataset release      
         - Audio features: VGGish 
         - Visual features: I3D 
         
### - Baseline system release
      The system release is scheduled on July 2oth
      *You can find a setup using the prototype data and the given audio and visual features: 
                        <https://arxiv.org/abs/1806.08409>

## - Track Description
Welcome to the Audio Visual Scene-Aware Dialog (AVSD) challenge and dataset. This challenge is one track of  the **7th Dialog System Technology Challenges (DSTC7) workshop.**
The task is to build a system that generates responses in a dialog about an input video.

### - Tasks

In this challenge, the system must generate responses to a user input in the context of a given dialog.  
This context consists of a dialog history (previous utterances by both user and system) in addition to video and audio information that comprise the scene. 
The quality of a system’s automatically generated sentences is evaluated using objective measures to determine whether or not the generated responses are natural and informative.

#### 1. Task 1: Video and Text 
    a. Use the video and text training data provided but no external data sources, 
       other than publicly available pre-trained feature extraction models.

    b. External data may also be used for training.

#### 2. Task 2: Text Only 
    a. Do not use the input videos for training or testing. 
       Use only the text training data (dialogs and video descriptions) provided. 
    b. Any publicly available text data may also be used for training.

### - Dataset
#### Proto type data set:

|               |   Training    |  Validation   |     Test      |
| ------------- | ------------- | ------------- | ------------- |
| # of Dialogs  |     6172      |      732      |      733      |
| # of Turns    |    123,480    |     14,680    |     14,660    |
| # of Words    |    1,163,969  |    138,314    |    138,790    |


### - Contact Information

halamri3@gatech.edu & chori@merl.com
