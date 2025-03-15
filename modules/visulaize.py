import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_results(Y_test, X_test_prediction):
    positive_tweets = sum(1 for i in Y_test if i == 1)
    negative_tweets = sum(1 for i in Y_test if i == 0)
    total_tweets = positive_tweets + negative_tweets
    positive_percentage = (positive_tweets / total_tweets) * 100
    negative_percentage = (negative_tweets / total_tweets) * 100

    labels = ['Positive', 'Negative']
    sizes = [positive_percentage, negative_percentage]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.title('Sentiment Analysis Results')
    plt.show()

    cm = confusion_matrix(Y_test, X_test_prediction)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()