from helper.lib import *


def plot_dataset(X, y, theta = []):
    plt.figure(figsize=(8, 8))
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', c='b',  label='Positive')
    plt.scatter(X[y == 0, 0], X[y == 0, 1],marker='x', c='r', label='Negative')

    if len(theta) > 0:
        x_line = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
        y_line = -(theta[0] + theta[1] * x_line) / theta[2]
        plt.plot(x_line, y_line, color='black', linestyle='-', label='Classification Line')

def plot_loss(loss):
    plt.figure(figsize=(8, 8))
    plt.plot(loss)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs Iteration')
    plt.show()

def plot_accuracy(acc):
    plt.figure(figsize=(8, 8))
    plt.plot(acc)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Iteration')
    plt.show()

def plot_heatmap(cnf_matrix):
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')