import pandas as pd
import pickle
import string
import matplotlib.pyplot as plt


# Load the tokenized dataframe from the pickle file
with open('D:/research/llm_impact_on_early_stage_design_ideation/on_campus_students/experiment_2/tokenized_df_of_prompts.pkl', 'rb') as f:
    tokenized_df_with_empt_strings = pickle.load(f)

print(tokenized_df_with_empt_strings)

# create a list of lists containing the non-empty tokens in each row
token_lists = []
for index, row in tokenized_df_with_empt_strings.iterrows():
    token_list = [token for token in row if isinstance(token, list) and token != '']
    token_lists.append(token_list)

# print(token_lists)

flat_token_lists = []

for row in token_lists:
    # print(f'now in row {row}')
    for sublist in row:
        # print(f'isolated {sublist}')
        flat_token_lists.append(sublist)

# print(flat_token_lists)

# Removing duplicates
flat_token_lists_no_duplicates = list(phrase.split(" ") for phrase in set(list(" ".join(list_of_words) for list_of_words in flat_token_lists)))


# Removing punctuation marks from each sublist and reinitialize flat_token_list_no_duplicates

tokens = [[token for token in sublist if token not in string.punctuation] for sublist in flat_token_lists_no_duplicates]
flat_token_lists_no_duplicates = tokens

# print(flat_token_lists_no_duplicates)

# function that converts the list of tokens into vector word embeddings
def embedding_generator(flat_token_lists):
    import torch
    from transformers import DistilBertModel, DistilBertTokenizer
    # Suppressing the warning from distilBERT
    import logging
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    # Load pre-trained DistilBERT model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    # Tokenize the corpus
    tokenized_corpus = [tokenizer.tokenize(" ".join(tokens)) for tokens in flat_token_lists]

    # Convert tokens to IDs
    input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_corpus]

    # Pad sequences to the same length
    max_len = max([len(seq) for seq in input_ids])
    input_ids = [seq + [tokenizer.pad_token_id] * (max_len - len(seq)) for seq in input_ids]

    # Convert inputs to PyTorch tensors
    input_ids = torch.tensor(input_ids)

    # Generate embeddings
    with torch.no_grad():
        embeddings = model(input_ids)[0][:, 0, :].numpy()
    return embeddings

embeddings = embedding_generator(flat_token_lists_no_duplicates)

# The embeddings will be a 2D array with shape (num_sentences, embedding_size)
# print(embeddings)

print(f'the length of flat_token_lists_no_duplicates is {len(flat_token_lists_no_duplicates)} and the length of embeddings is {len(embeddings)}')


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Initialize TSNE
tsne = TSNE(n_components=2, random_state=42)

# Fit and transform the embeddings to 2D space
embeddings_2d = tsne.fit_transform(embeddings)

# finding the optimal number of clusters using the silouhette score parameter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def get_optimal_num_clusters(embeddings):
    max_num_clusters = min(len(embeddings), 10)
    num_clusters_range = range(2, max_num_clusters+1)
    sil_scores = []
    for num_clusters in num_clusters_range:
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
        kmeans.fit(embeddings)
        labels = kmeans.labels_
        sil_score = silhouette_score(embeddings, labels)
        sil_scores.append(sil_score)
    optimal_num_clusters = num_clusters_range[np.argmax(sil_scores)]
    return optimal_num_clusters
optimal_number_of_clusters = get_optimal_num_clusters(embeddings_2d)

print(f'The optimal number of clusters is {optimal_number_of_clusters}')

# Performing the clustering using K means
k = 2
# Initialize KMeans model with k clusters
kmeans = KMeans(n_clusters=k,n_init=10)

# Fit the model to the embeddings
kmeans.fit(embeddings)

# Get the cluster labels
labels = kmeans.labels_

def plot_clusters(embeddings_2d):
    # x_embedded and y_embedded are the TSNE embeddings
    x_embedded = embeddings_2d[:,0]
    y_embedded = embeddings_2d[:,1]
    # words_list is the list of words corresponding to each embedding
    words_list = [' '.join(tokens) for tokens in flat_token_lists_no_duplicates]

    # Scatter plot of the embeddings
    plt.figure(figsize=(10, 10))
    for i in range(len(set(labels))):
        x = x_embedded[labels == i]
        y = y_embedded[labels == i]
        color = {0:'red',1:'blue',2:'green'}
        plt.scatter(x, y, c=color[i])
        
    # Add annotations for each point
    for i, word in enumerate(words_list):
        x = x_embedded[i]
        y = y_embedded[i]
        label = labels[i]
        plt.annotate(word, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', fontsize=8)

    # Add legends and titles
    plt.title('TSNE Visualization of Word Embeddings')
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')
    plt.legend(['Cluster 1', 'Cluster 2', 'Cluster 3'], loc='best')

    plt.show()

plot_clusters(embeddings_2d)
