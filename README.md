# PyTorch Sentiment Analysis – Deep Learning & Transformers

## 📌 Project Overview
This project implements an **advanced sentiment analysis system** using **PyTorch**, progressing from baseline neural models to **Transformer-based architectures (BERT)**.  
The objective is to understand and compare how different deep learning models capture semantic meaning in text and improve sentiment classification accuracy.

The project demonstrates a **complete NLP pipeline**: data preprocessing, tokenization, model building, training, evaluation, and inference.

---

## 🎯 Problem Statement
User-generated text such as movie reviews contains rich sentiment information, but understanding context and long-term dependencies is challenging for traditional models.

This project addresses:
- How different neural architectures perform on sentiment classification
- How **context-aware embeddings (BERT)** improve performance
- How to build scalable NLP solutions using PyTorch

---

## 🧠 Models Implemented
The project incrementally builds and evaluates the following models:

1. **Neural Bag of Words (Baseline)**
2. **Recurrent Neural Networks (LSTM / GRU)**
3. **Convolutional Neural Networks (CNN) for Text**
4. **Transformer-based Model (BERT + Bi-directional GRU)**

Each model is trained and evaluated on the **IMDB movie reviews dataset**.

---

## 🏗️ Transformer-Based Architecture
- **Pretrained BERT (bert-base-uncased)** used as embedding layer
- Transformer weights are **frozen** for efficient training
- **Bi-directional GRU** learns task-specific representations
- Binary sentiment prediction using **BCEWithLogitsLoss**

This approach balances **high accuracy with reduced training cost**.

---

## 📊 Results
| Model | Validation Accuracy |
|------|--------------------|
| Neural BoW | ~84% |
| LSTM / GRU | ~88% |
| CNN | ~89% |
| **BERT + BiGRU** | **~92%** |

✔ Achieved **~91.5% test accuracy** using the Transformer-based model.

---

## 🔬 Dataset
- **IMDB Movie Reviews Dataset**
- 50,000 labeled reviews (positive / negative)
- Balanced dataset
- Tokenized using **BERT tokenizer**

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Deep Learning:** PyTorch  
- **NLP:** TorchText, HuggingFace Transformers  
- **Models:** LSTM, GRU, CNN, BERT  
- **Evaluation:** Accuracy, Loss  
- **Environment:** Jupyter Notebook  

---

## 📂 Project Structure

<img width="584" height="450" alt="image" src="https://github.com/user-attachments/assets/e8582bc5-bb73-40e8-9a70-9b94c2964add" />


This repo contains tutorials covering understanding and implementing sequence classification models using [PyTorch](https://github.com/pytorch/pytorch), with Python 3.9. Specifically, we'll train models to predict sentiment from movie reviews.

**If you find any mistakes or disagree with any of the explanations, please do not hesitate to [submit an issue](https://github.com/bentrevett/pytorch-sentiment-analysis/issues/new). I welcome any feedback, positive or negative!**

## Getting Started

Install the required dependencies with: `pip install -r requirements.txt --upgrade`.

## Tutorials

-   1 - [Neural Bag of Words](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/main/1%20-%20Neural%20Bag%20of%20Words.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/main/1%20-%20Neural%20Bag%20of%20Words.ipynb)

    This tutorial covers the workflow of a sequence classification project with PyTorch. We'll cover the basics of sequence classification using a simple, but effective, neural bag-of-words model, and how to use the datasets/torchtext libaries to simplify data loading/preprocessing.

-   2 - [Recurrent Neural Networks](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/main/2%20-%20Recurrent%20Neural%20Networks.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/main/2%20-%20Recurrent%20Neural%20Networks.ipynb)

    Now we have the basic sequence classification workflow covered, this tutorial will focus on improving our results by switching to a recurrent neural network (RNN) model. We'll cover the theory behind RNNs, and look at an implementation of the long short-term memory (LSTM) RNN, one of the most common variants of RNN.

-   3 - [Convolutional Neural Networks](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/main/3%20-%20Convolutional%20Neural%20Networks.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/main/3%20-%20Convolutional%20Neural%20Networks.ipynb)

    Next, we'll cover convolutional neural networks (CNNs) for sentiment analysis. This model will be an implementation of [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882).

-   4 - [Transformers](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/main/4%20-%20Transformers.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/main/4%20-%20Transformers.ipynb)

    Finally, we'll show how to use the transformers library to load a pre-trained transformer model, specifically the BERT model from [this](https://arxiv.org/abs/1810.04805) paper, and use it for sequence classification.




## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Swamy12S/PyTorch-Sentiment-Analysis.git
cd PyTorch-Sentiment-Analysis
2️⃣ Install dependencies
pip install -r requirements.txt

