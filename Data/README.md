# Audio Video Scene Aware Dialog Dataset V 0.1 

There are 3 types of data such as text, audio, visual data. 

The text data contains the 10 sets of QAs/video  and 5 descriptions/video (now collecting).

You can get the data from the following site:

https://drive.google.com/drive/u/2/folders/1JGE4eeelA0QBA7BwYvj89kSClE3f9k65

## Text data
- This text-based human dialog data for video from Charades Dataset (training, testing and validation): http://allenai.org/plato/charades/
- Each dialog consists of 10 round of questions/answeres. 

#### - Relevan files:

   * README.txt    
   * videoDial_train_v01.json : Annotations for the trainign set.   
   * videoDial_test_v01.json:   Annotations for the testing set.
  
  
#### - Annotations Files format:  

     { 
     "Dialogs": [  
         "Image_id" : ""YSE1G", 
         "Summary": "the girl walks into a room with a dog with a towel around her neck . she does some stretches and then drops the towel ",
         "Caption": "a person walked through a doorway into the living room with a towel draped around their neck , and closed the door . the person stretched and threw the towel on the floor."  
         "Dialog": [  
         {    
               " Question": "is there only one person ?"
               " Answer": "there is only one person and a dog .",   
          },   
          {
                " Question 2": ....
                " Answer 2": .....
          ..
          ..
          }
                " Question 10": ....
                " Answer 10": .....
          } 
          ] 
    }

## Audio data
    Audio features are extracted using the VGGish model.
      S. Hershey, S. Chaudhuri, D. P. W. Ellis, J. F. Gemmeke, A. Jansen, R. C. Moore, M. Plakal,
      D. Platt, R. A. Saurous, B. Seybold, M. Slaney, R. J. Weiss, and K. Wilson, “CNN architectures
      for large-scale audio classification,” in ICASSP, 2017.
    
    You can download the files from the following link:

## Visual data
    Visal features are extracted using the I3D model.
      Joao Carreira and Andrew Zisserman, “Quo vadis, action recognition? a new model and the
      kinetics dataset,” in CVPR, 2017.

    You can download the files from the following link:
