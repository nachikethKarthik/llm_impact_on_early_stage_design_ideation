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

# Split user IDs into groups (assuming 6 groups)
num_rows = 2
num_cols = 3
num_plots = num_rows * num_cols
user_groups = [[] for _ in range(num_plots)]
users = list(set(user_ids))

for i, user_id in enumerate(users):
    group_idx = i % num_plots
    user_groups[group_idx].append(user_id)

# Create subplots with 2 rows and 3 columns
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
colors = plt.cm.get_cmap('tab20b', len(user_ids))
for group_idx, user_group in enumerate(user_groups):
    row_idx = group_idx // num_cols
    col_idx = group_idx % num_cols
    ax = axs[row_idx, col_idx]
    ax.set_title(f'Group {group_idx+1}')

    for i, user_id in enumerate(user_group):
        indices = [idx for idx, uid in enumerate(user_ids) if uid == user_id]
        x = reduced_vectors[indices, 0]
        y = reduced_vectors[indices, 1]
        color = colors(i)

        for j, prompt in enumerate(user_data[user_id]):
            ax.scatter(x[j], y[j], color=color, s=3)
            ax.text(x[j], y[j], f'{j+1}: {prompt}', fontsize=5)

            if j > 0:
                ax.arrow(x[j-1], y[j-1], x[j] - x[j-1], y[j] - y[j-1],
                         length_includes_head=True, head_width=0.03, head_length=0.05,
                         color=color)

# Create a common legend for all plots
legend_labels = [plt.scatter([], [], color=colors(i), label=user_id) for i, user_id in enumerate(set(user_ids))]
# plt.legend(handles=legend_labels, title='User ID')

# plt.tight_layout()
plt.show()
