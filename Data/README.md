# Audio Video Scene Aware Dialog Dataset V 0.1 

- This text-based human dialog data for video from Charades Dataset (training, testing and validation): http://allenai.org/plato/charades/
- Each dialog consists of 10 round of questions/answeres. 

#### - Relevan files:

   * README.txt    
   * videoDial_train_v01.json : Annotations for the trainign set.   
   * videoDial_test_v01.json:   Annotations for the testing set.
  
  
#### - Annotations Files format:  

  { 
      "dialogs": [  
         "image_id" : ""YSE1G", 
         "summary": "the girl walks into a room with a dog with a towel around her neck . she does some stretches and then drops the towel ",   
         "dialog": [  
         {  
               "answer": "there is only one person and a dog .",   
               "question": "is there only one person ?"  
          },   
          ..
          ..
          ..
          }

    }


  


         
    



