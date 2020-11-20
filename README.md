# Run the program
To execute the program run this command you can run the jupyter notebook only and you will get the results, in this file we used rule-based tagger and also our pre-trained sequence model to output the results.



In case you want to train the model you can run sequence_tagger.py file by the following command:


python sequence_tagger.py


and for more option you can use --help command to see more options of training.

-i for number of iteration, we set 10 iterations as default value
-l to load pre-trained Spacy model, default value is False (True to load the model)
-o the path to save the trained model, default value is 'ouput_dir' directory in the same work directory



 





# Astronomy termonology:
articleTermonoly.tsv is the war words from our corpus
astronomy.xls is  the list of astronomy terms preprocessed and selected.



