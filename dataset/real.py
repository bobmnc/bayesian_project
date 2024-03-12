from datasets import load_dataset
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def count_words(review_tokens):
    word_freq = {}
    for word in review_tokens:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    return word_freq

def preprocess_review(review,stop_words):
    review = re.sub(r'\W+', ' ', review)
    review = review.lower()
    tokens = word_tokenize(review)
    
    
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

def real_dataset():
    dataset = load_dataset("ag_news",split='train')
    
    df = dataset.to_pandas()
    dataset_test = load_dataset("ag_news",split='test')
    df2 = dataset_test.to_pandas()
    
    df = pd.concat([df,df2])
    
    df = dataset.to_pandas()
    
    
    
    # Preprocess the reviews
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    df['words'] = df['text'].apply(lambda text : preprocess_review(text,stop_words))

    # Use multiprocessing to count word frequencies
    with Pool() as p:
        word_freqs = p.map(count_words, df['words'])
    
    # Combine the word frequencies
    all_word_freqs = {}
    for word_freq in word_freqs:
        for word, freq in word_freq.items():
            if word in all_word_freqs:
                all_word_freqs[word] += freq
            else:
                all_word_freqs[word] = freq
    
    # Create a numpy array from the word frequencies
    word_freq_array = np.sort(np.array(list(all_word_freqs.values())))[::-1]

    return word_freq_array