# Audio Video Scene Aware Dialog Dataset V 0.1 

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


  


         
    



