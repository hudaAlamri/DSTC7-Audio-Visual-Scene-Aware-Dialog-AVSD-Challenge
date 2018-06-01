# Audio Video Scene Aware Dialog Dataset V 0.1 

- This text-based human dialog data for video from Charades Dataset (training, testing and validation): http://allenai.org/plato/charades/
- Each dialog consists of 10 round of questions/answeres. 

#### - Relevan files:

   * README.txt    
   * videoDial_train_v01.json : Annotations for the trainign set.   
   * videoDial_test_v01.json:   Annotations for the testing set.
  
  
#### - Annotations Files format:  

  * Id: video identifier - the original charades video Ids.     
  * Script: the human-generated scripts, from Charades dataset.        
  * Split : "Train", "Test"      
  * Dial_id : unique identifer for each dialog in the dataset.  
    		-	AQ_id : the order ot the question/answer pair in the dailog. (1-10)     
        -	Question : One of 10 questions in this round of dialog.     
        -	Answer :   The corresponding answer to AQ_id question  
        - Summary:   A description of the video written by the answerer.  
              
  


         
    



