# NLP Coursework 2023
  
## Project repository

This project aims to develop a large language model (LLM) to predict whether a text contains patronising or condescending language (PCL) for task 4- 1 in the SemEval 2022 competition. 

For development process and summary of insights, see ```NLP_report.pdf```.


## In this project
- We used one of the pre-trained LLMs, DeBERTa, from Hugging Face and fine-tuned it for our custom binary classification task.
- We conducted explorary data analysis and visualisation on the highly class-imbalanced language corpus.
- The fine-tuned DeBERTa achieved a final F1 score of 0.61, which exceeds the RoBERTa-base score of 0.48. The corresponding accuracy is 0.93.
- We created two simpler language models, Bag of Words (BoW) and Multi-layer Perceptron (MLP) and compared their performance with the DeBERTa.
