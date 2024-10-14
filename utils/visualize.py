import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.manifold import TSNE

from utils.func import calculate_frechet_distance


def score_dist(y_label, y_score):
    data = pd.DataFrame({'Score': y_score, 'Label': y_label})
    sns.kdeplot(data=data, x='Score', hue='Label', fill=True)
    plt.show()


def plot_two_rel(data_x, data_y, name_x, name_y, ax=None, set_axis_labels=False):
    if ax is None:
        fig, ax = plt.subplots()
        
    # Create the scatter plot
    ax.scatter(data_x, data_y, alpha=0.7)
    if set_axis_labels:
        ax.set_xlabel(name_x)
        ax.set_ylabel(name_y)

    # Calculate the linear regression
    slope, intercept, r_value, p_value, std_err = linregress(data_x, data_y)

    # Plot the regression line
    x = np.linspace(min(data_x), max(data_x), 100)
    y = slope * x + intercept
    ax.plot(x, y, color='red', label=f'Linear fit: $R^2$={r_value**2:.2f}')
    ax.plot([min(data_x), max(data_x)], [min(data_x), max(data_x)], color='gray', linestyle='--', label='y=x')

    ax.legend()
    ax.grid(True)


def tsne_show_labels(data, labels):
    tsne = TSNE(n_components=2, random_state=0, perplexity=5)
    tsne_results = tsne.fit_transform(data)

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette='viridis')
    plt.title('t-SNE Visualization of Hidden Vectors')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()


# def draw_task(root_hs, task_types, task_data):
#     x_dist, y_acc = [], []

#     hs1 = root_hs
#     mu1 = np.mean(hs1, axis=0)

#     for task in task_types:
#         print(f"Task: {task}", end=" | ")
#         hs2 = task_data[task]['hidden_states']
#         mu2 = np.mean(hs2, axis=0)
#         diff = mu1 - mu2

#         # Calculate the Fréchet distance
#         distance = np.linalg.norm(diff)
#         x_dist.append(distance)

#         # Calculate the delta accuracy
#         y_acc.append(task_data[task]['acc'])

#         print(f"Dist: {distance:.2f} | Acc: {task_data[task]['acc']:.2f}")

#     # Plot the relationship between Fréchet distance and delta accuracy
#     ax = plt.gca()
#     plot_two_rel(x_dist, np.abs(y_acc), "Distance from the train set", "Accuracy", ax=ax, set_axis_labels=True)
#     plt.show()
    
    
# def draw_task(root_hs, task_types, task_data):
#     x_dist, y_acc = [], []

#     hs1 = root_hs
#     mu1 = np.mean(hs1, axis=0)
#     sigma1 = np.cov(hs1, rowvar=False)

#     for task in task_types:
#         print(f"Task: {task}", end=" | ")
#         hs2 = task_data[task]['hidden_states']
#         mu2 = np.mean(hs2, axis=0)
#         sigma2 = np.cov(hs2, rowvar=False)

#         # Calculate the Fréchet distance
#         distance = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
#         x_dist.append(distance)

#         # Calculate the delta accuracy
#         y_acc.append(task_data[task]['acc'])

#         print(f"Dist: {distance:.2f} | Acc: {task_data[task]['acc']:.2f}")

#     # Plot the relationship between Fréchet distance and delta accuracy
#     ax = plt.gca()
#     plot_two_rel(x_dist, np.abs(y_acc), "Distance from the train set", "Accuracy", ax=ax, set_axis_labels=True)
#     plt.show()
    

def draw_task(task_types, task_data, ax=None):
    x_dist, y_acc = [], []

    for task in task_types:
        # print(f"Task: {task}", end=" | ")
        hs2 = task_data[task]['metric']
        mu2 = np.mean(hs2)
        distance = mu2 * 100
        
        # Calculate the Fréchet distance
        x_dist.append(distance)

        # Calculate the delta accuracy
        y_acc.append(task_data[task]['acc'])

        # print(f"Confidence: {distance:.2f} | Acc: {task_data[task]['acc']:.2f}")

    # Plot the relationship between Fréchet distance and delta accuracy
    if ax is None:
        ax = plt.gca()
    plot_two_rel(x_dist, np.abs(y_acc), "Confidence", "Accuracy", ax=ax, set_axis_labels=True)