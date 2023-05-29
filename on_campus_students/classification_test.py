
mega_list = ['cat', 'dog', 'elephant', 'giraffe', 'lion', 'zebra', 'monkey', 'tiger', 'kangaroo', 'panda', 'rhinoceros', 'hippopotamus', 'penguin', 'seagull', 'dolphin', 'whale', 'shark', 'crocodile', 'turtle', 'snake','apple', 'banana', 'orange', 'strawberry', 'blueberry', 'grape', 'kiwi', 'watermelon', 'mango', 'peach', 'pineapple', 'lemon', 'lime', 'pear', 'plum', 'raspberry', 'pomegranate', 'cherry', 'coconut', 'fig','red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'gray', 'black', 'white', 'maroon', 'teal', 'navy', 'beige', 'turquoise', 'lavender', 'peach', 'mustard', 'olive','doctor', 'teacher', 'engineer', 'lawyer', 'accountant', 'dentist', 'nurse', 'architect', 'scientist', 'artist', 'musician', 'chef', 'writer', 'programmer', 'athlete', 'entrepreneur', 'psychologist', 'economist', 'philosopher', 'politician']

animals = ['cat', 'dog', 'elephant', 'giraffe', 'lion', 'zebra', 'monkey', 'tiger', 'kangaroo', 'panda', 'rhinoceros', 'hippopotamus', 'penguin', 'seagull', 'dolphin', 'whale', 'shark', 'crocodile', 'turtle', 'snake']
fruits = ['apple', 'banana', 'orange', 'strawberry', 'blueberry', 'grape', 'kiwi', 'watermelon', 'mango', 'peach', 'pineapple', 'lemon', 'lime', 'pear', 'plum', 'raspberry', 'pomegranate', 'cherry', 'coconut', 'fig']
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'gray', 'black', 'white', 'maroon', 'teal', 'navy', 'beige', 'turquoise', 'lavender', 'peach', 'mustard', 'olive']
professions = ['doctor', 'teacher', 'engineer', 'lawyer', 'accountant', 'dentist', 'nurse', 'architect', 'scientist', 'artist', 'musician', 'chef', 'writer', 'programmer', 'athlete', 'entrepreneur', 'psychologist', 'economist', 'philosopher', 'politician']

def embedding_generator(mega_list):
    import torch
    from transformers import DistilBertModel, DistilBertTokenizer
    # Suppressing the warning from distilBERT
    import logging
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    # Load pre-trained DistilBERT model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    # Tokenize the corpus
    tokenized_corpus = [tokenizer.tokenize(word) for word in mega_list]

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

embeddings = embedding_generator(mega_list)

print(f'the length of mega_list is {len(mega_list)} and the length of embeddings is {len(embeddings)} and size of each embedding is {len(embeddings[0])}')
print(embeddings[0])

from sklearn.manifold import TSNE
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
k = optimal_number_of_clusters
# Initialize KMeans model with k clusters
kmeans = KMeans(n_clusters=k,n_init=10)

# Fit the model to the embeddings
kmeans.fit(embeddings)

# Get the cluster labels
labels = kmeans.labels_



import matplotlib.pyplot as plt

def plot_clusters(embeddings_2d):
    import matplotlib.pyplot as plt

    # x_embedded and y_embedded are the TSNE embeddings
    x_embedded = embeddings_2d[:,0]
    y_embedded = embeddings_2d[:,1]
    # words_list is the list of words corresponding to each embedding
    words_list = mega_list
    # labels is the list of cluster labels corresponding to each embedding
    labels = kmeans.labels_

    # Scatter plot of the embeddings
    plt.figure(figsize=(10, 10))
    for i, word in enumerate(words_list):
        x = x_embedded[i]
        y = y_embedded[i]
        label = labels[i]
        color = {0:'red',1:'blue',2:'green',3:'yellow',4:'black'}
        plt.scatter(x, y, c=color[label])
        plt.annotate(word, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', fontsize=8)

    # Add legends and titles
    plt.title('TSNE Visualization of Word Embeddings')
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')
    # plt.legend(['Cluster 1','Cluster 2','Cluster 3', 'Cluster 4','Cluster 5'], loc='best')

    plt.show()

plot_clusters(embeddings_2d)
