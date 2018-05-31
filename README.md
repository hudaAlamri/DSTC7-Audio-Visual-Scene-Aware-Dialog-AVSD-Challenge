# DSTC7-Audio-Visual-Scene-Aware-Dialog-AVSD-Challenge

# - Track Description
Welcome tothe Audio Visual Scene-Aware Dialog (AVSD) challenge and dataset. This challenge, is one track of the 7th Dialog System Technology Challenges (DSTC7) workshop.
The task is to build asystem that generates responses in a dialog about an input video.

## - Tasks

In this challenge, the system must generate responses toa user input in the context of a given dialog.  
This contextconsists of a dialog history (previous utterances by both userand system) in addition to video and audio information thatcomprise the scene. 
The quality of a system’s automaticallygenerated sentences is evaluated using objective measures to determine whether or not the generated responses are natural and informative.

### 1. Task 1: Video and Text 
   - Use the video and text training data provided but no external data sources, other than
      publicly available pre-trained feature extraction models.

   - External data may also be used for training.

### 2. Task 2: Text Only 
   - Do not use the input videos for training or testing. 
      Use only the text training data (dialogs and video descriptions) provided. 
   - Any publicly available text data may also be used for training.

## - Dataset


|               |   Training    |  Validation   |     Test      |
| ------------- | ------------- | ------------- | ------------- |
| # of Dialogs  |     7043      |      732      |      733      |
| # of Turns    |    123,480    |     14,680    |     14,660    |
| # of Words    |    1,163,969  |    138,314    |    138,790    |

