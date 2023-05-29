import pandas as pd
import pickle
import string
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# load the dataframe having the ideas with stopwords present for plotting
with open('D:/research/llm_impact_on_early_stage_design_ideation/on_campus_students/experiment_3/tokenized_df_of_ideas_exp_3_with_stopwords.pkl', 'rb') as f:
    tokenized_df_with_stopwords = pickle.load(f)
# print(tokenized_df_with_stopwords['0'].tolist())
# Load the tokenized dataframe from the pickle file
with open('D:/research/llm_impact_on_early_stage_design_ideation/on_campus_students/experiment_3/tokenized_df_of_ideas_exp_3.pkl', 'rb') as f:
    tokenized_df_with_empt_strings = pickle.load(f)

# print(tokenized_df_with_empt_strings)

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

# Removing duplicate ideas
flat_token_lists_no_duplicates = list(phrase.split(" ") for phrase in set(list(" ".join(list_of_words) for list_of_words in flat_token_lists)))
idea_list_with_stopwords_no_duplicates = list(set(tokenized_df_with_stopwords['0'].tolist()))
# Removing punctuation marks from each sublist and reinitialize flat_token_list_no_duplicates
# tokens = [[token for token in sublist if token not in string.punctuation] for sublist in flat_token_lists_no_duplicates]
# flat_token_lists_no_duplicates = tokens

# print(flat_token_lists_no_duplicates)

# function that converts the list of tokens into vector word embeddings
def embedding_generator(flat_token_lists_no_duplicates):
    # List of ideas
    ideas = [" ".join(sublist) for sublist in flat_token_lists_no_duplicates] 
    # print(ideas)
    # Tokenize the ideas into a list of lists of words
    tokenized_ideas = [idea.split() for idea in ideas]

    # Train the Word2Vec model
    model = Word2Vec(tokenized_ideas, vector_size=100, window=5, min_count=1, workers=4)

    # Function to get the idea representation
    def get_idea_vector(idea):
        # Tokenize the idea
        tokens = idea.split()

        # Initialize an empty vector
        idea_vector = np.zeros(model.vector_size)

        # Count the number of tokens
        count = 0

        # Calculate the sum of word vectors for each token in the idea
        for token in tokens:
            if token in model.wv:
                idea_vector += model.wv[token]
                count += 1

        # Divide the sum by the count to get the average
        if count > 0:
            idea_vector /= count

        return idea_vector

    # Get the vector representation for each idea
    idea_vectors = [get_idea_vector(idea) for idea in ideas]

    return idea_vectors

embeddings = embedding_generator(flat_token_lists_no_duplicates)
# convert embeddings to a numpy array
embeddings = np.array(embeddings)
# print(embeddings)

print(f'the length of flat_token_lists_no_duplicates is {len(flat_token_lists_no_duplicates)} and the length of embeddings is {len(embeddings)}')


# Initialize TSNE
tsne = TSNE(n_components=2, random_state=42)

# Fit and transform the embeddings to 2D space
embeddings_2d = tsne.fit_transform(embeddings)

# finding the optimal number of clusters using the silouhette score parameter

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
k = 9
# Initialize KMeans model with k clusters
kmeans = KMeans(n_clusters=k,n_init=10)

# Fit the model to the embeddings
kmeans.fit(embeddings_2d)

# Get the cluster labels
labels = kmeans.labels_


import matplotlib.pyplot as plt

def plot_clusters(embeddings_2d):

    # x_embedded and y_embedded are the TSNE embeddings
    x_embedded = embeddings_2d[:,0]
    y_embedded = embeddings_2d[:,1]
    # words_list is the list of words corresponding to each embedding
    words_list = [' '.join(tokens) for tokens in flat_token_lists_no_duplicates]
    # idea_list_with_stopwords_no_duplicates
    # labels is the list of cluster labels corresponding to each embedding
    labels = kmeans.labels_

    # Scatter plot of the embeddings
    plt.figure(figsize=(10, 10))
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow','black', 'purple', 'gray']
    # , ,'darkgray', 'maroon', 'olive', 'navy',, 'teal', 'aqua', 'lime', 'fuchsia', 'silver' 'white', 'lightgray'
    for i in range(len(colors)):
        x = x_embedded[labels==i]
        y = y_embedded[labels==i]
        plt.scatter(x, y, c=colors[i], label='Cluster {}'.format(i+1), alpha=0.6)
    for i, word in enumerate(words_list):
        plt.annotate(word, xy=(x_embedded[i], y_embedded[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', fontsize=6,alpha = 0.8)
    
    # Cluster centers
    cluster_centers = kmeans.cluster_centers_
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=100)
    
    # Connect cluster centers with lines and display distances
    for i in range(k):
        x = [cluster_centers[i, 0]]
        y = [cluster_centers[i, 1]]
        plt.plot(x, y, color='gray', linestyle='-', alpha=0.5)
        for j in range(i + 1, k):
            line_length = np.sqrt((cluster_centers[i, 0] - cluster_centers[j, 0]) ** 2 + (cluster_centers[i, 1] - cluster_centers[j, 1]) ** 2)
            plt.plot([cluster_centers[i, 0], cluster_centers[j, 0]], [cluster_centers[i, 1], cluster_centers[j, 1]], color='gray', linestyle='--', alpha=0.5)
            plt.text((cluster_centers[i, 0] + cluster_centers[j, 0]) / 2, (cluster_centers[i, 1] + cluster_centers[j, 1]) / 2, f"{line_length:.2f}", color='black', fontsize=8,alpha=1)

    # Add legends and titles
    plt.title('TSNE Visualization of Word Embeddings')
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')
    plt.legend(loc='best')

    plt.show()

plot_clusters(embeddings_2d)