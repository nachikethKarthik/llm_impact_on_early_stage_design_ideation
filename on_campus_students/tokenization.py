import nltk
import numpy as np
import pickle

# nltk.download('punkt')

import pandas as pd

df = pd.read_csv('exp_3_processed_idea_only.csv')
df_with_stopwords = df
from nltk.tokenize import word_tokenize

def custom_tokenizer(text):
    # Tokenize the text using the NLTK word_tokenize function
    
    # print(f'Now tokenizing {text} and its type is {type(text)}')

    # If the cell is empty ignore it and move to next cell
    if (type(text) == float):
        return ''
    tokens = word_tokenize(text)

    # Join together any phrases made up of multiple words into a single token
    phrases = []
    i = 0
    while i < len(tokens):
        if ' ' in tokens[i]:
            # If the token contains a space, it is a phrase made up of multiple words
            phrase = tokens[i]
            i += 1
            while i < len(tokens) and ' ' in tokens[i]:
                phrase += ' ' + tokens[i]
                i += 1
            phrases.append(phrase)
        else:
            # Otherwise, it is a single-word token
            phrases.append(tokens[i])
            i += 1

    return phrases

tokenized_df = df.applymap(custom_tokenizer)

# print(tokenized_df)

# Removing stop-words
from nltk.corpus import stopwords
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

df_without_stopwords = tokenized_df.applymap(lambda x: [word for word in x if word not in stop_words] if isinstance(x, list) and x else x)

print(df_without_stopwords)

with open('tokenized_df_of_ideas_exp_3.pkl', 'wb') as f:
    pickle.dump(df_without_stopwords, f)
with open('tokenized_df_of_ideas_exp_3_with_stopwords.pkl', 'wb') as f:
    pickle.dump(df_with_stopwords, f)
