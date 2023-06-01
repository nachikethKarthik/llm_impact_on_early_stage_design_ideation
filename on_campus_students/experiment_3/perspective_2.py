from csv_processing import user_data

import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize

# Tokenize the prompts
tokenized_data = []
for user_id, prompts in user_data.items():
    for prompt in prompts:
        tokens = word_tokenize(prompt)
        tokenized_data.append(tokens)

# Train the Word2Vec model
model = Word2Vec(tokenized_data, min_count=1)

# Get the word vectors
word_vectors = model.wv

# Create a list to store the word vectors
vectors = []
user_ids = []  # Keep track of user IDs
for user_id, prompts in user_data.items():
    user_ids.extend([user_id] * len(prompts))
    for prompt in prompts:
        tokens = word_tokenize(prompt)
        vector = sum([word_vectors[word] for word in tokens]) / len(tokens)
        vectors.append(vector)

# Convert vectors to a NumPy array
vectors = np.array(vectors)

# Apply TSNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create a scatter plot
plt.figure(figsize=(10, 8))
colors = plt.cm.get_cmap('tab20b', len(user_data))

for i, user_id in enumerate(set(user_ids)):
    indices = [idx for idx, uid in enumerate(user_ids) if uid == user_id]
    x = reduced_vectors[indices, 0]
    y = reduced_vectors[indices, 1]
    color = colors(i)

    for j, prompt in enumerate(user_data[user_id]):
        plt.scatter(x[j], y[j], color=color, s=3)
        plt.text(x[j], y[j], f'{j+1}: {prompt}', fontsize=5)

        if j > 0:
            plt.arrow(x[j-1], y[j-1], x[j] - x[j-1], y[j] - y[j-1],
                      length_includes_head=True, head_width=0.03, head_length=0.05,
                      color=color)

# Create a legend
legend_labels = [plt.scatter([], [], color=colors(i), label=user_id) for i, user_id in enumerate(set(user_ids))]
plt.legend(handles=legend_labels, title='User ID')

plt.title('Prompts Scatter Plot')
plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.show()
























