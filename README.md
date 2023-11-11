# NLP Coursework 2023
  
## Project repository

This project aims to develop a large language model (LLM) to predict whether a text contains patronising or condescending language (PCL) for task 4- 1 in the SemEval 2022 competition. 

One of the challenging in this project is that the given data is highly class-imbalanced: there are much more negative samples than positive ones (993 vs 9476).

For whole development process and summary, see "NLP_report.pdf".


## In this project
- We used one of the LLMs, DeBERTa, from Hugging face and fine-tuned it for our custom binary classification task.
- Conducted explorary data analysis and visualisation on the language corpus to understand the data.
- The fine-tuned DeBERTa achieved a final F1 score of 0.61, which exceeds the RoBERTa-base score of 0.48. The corresponding accuracy is 0.93.
- Created two simpler language models, Bag of Words (BoW) and Multi-layer Perceptron (MLP) and compared their performance with the DeBERTa.
