from LogisticRegression import LogsticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
from collections import Counter
import time

def preprocess_text(df, text_column):
    """
    This function preprocesses the text in the specified text column of a DataFrame and returns a new DataFrame
    that contains meaningful numerical vectors that represent the text.
    """
    # Define a list of stopwords to remove from the text
    stopwords = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its',
                 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with']

    # Create a list to store the preprocessed data
    preprocessed_data = []

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        # Get the text from the specified text column
        text = row[text_column]

        # Convert all text to lowercase
        text = text.lower()

        # Remove any URLs from the text
        text = re.sub(r'http\S+', '', text)

        # Remove any punctuation from the text
        text = re.sub(r'[^\w\s]', '', text)

        # Split the text into individual words
        words = text.split()

        # Remove any stopwords from the text
        words = [word for word in words if word not in stopwords]

        # Count the number of occurrences of each word in the text
        word_counts = Counter(words)

        # Add the row dictionary to the preprocessed data list
        preprocessed_data.append(word_counts)

    # Create a new DataFrame from the preprocessed data list
    new_df = pd.DataFrame(preprocessed_data)

    # Replace any missing values in the new DataFrame with 0
    new_df.fillna(0, inplace=True)

    # Return the new DataFrame
    return new_df
def draw_roc_curve(model, X_train, y_train):
    """
    Draw the Receiver Operating Characteristic (ROC) curve for a binary classification model.

    Parameters
    ----------
    model : object
        A logstic regression model with a predict_proba method.
    X_train : array-like
        Training data, where n_samples is the number of samples and n_features is the number of features.
    y_train : array-like
        Target values, where n_samples is the number of samples and n_classes is the number of classes.

    Returns
    -------
    float
        The optimal threshold value.
    """

    # Make list of thresholds
    list_thresholds = np.arange(0, 1.001, 0.001)

    # Make list of predicted probabilities
    y_prob = model.predict_proba(X_train)

    # Define list of scores per threshold
    dict_threshold = {}
    tpr_list = []
    fpr_list = []

    # Check score per threshold
    for threshold in list_thresholds:
        tpr, fpr = model.tpr_fpr(y_train, y_prob, threshold)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        dict_threshold[threshold] = tpr - fpr

    # Find the optimal threshold
    best_threshold = max(dict_threshold, key=lambda k: dict_threshold[k])

    # Print the optimal threshold
    print(f"Optimal Threshold: {best_threshold:.2f}")

    # Plot ROC curve
    plt.plot(fpr_list, tpr_list, "-o")

    # Plot line y=x
    plt.plot([0, 1], [0, 1], "--")

    # Add labels and title
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    # Show plot
    plt.show()

    return best_threshold


if __name__ == '__main__' :
    start_time = time.time()
    path = r"C:\Users\97252\Desktop\שיעורי בית\שנה ב\סמסטר ב\מבוא ללמידת מכונה\תרגיל 3\spam_ham_dataset.csv"
    df = pd.read_csv(path)
    targets = df['label_num']
    df=df.drop('label_num', axis=1)

    # Transfom the data
    df=preprocess_text(df,'text')

    # Split and shuffle
    X_train, X_test, y_train, y_test = train_test_split(df, targets, test_size=0.2,shuffle=False,random_state=42)

    # Add the intercept
    X_train = np.c_[X_train,np.ones(X_train.shape[0])]
    X_test=np.c_[X_test,np.ones(X_test.shape[0])]

    # Train and prediction
    model= LogsticRegression()
    model.fit(X_train,y_train)

    # Weights Of the model
    print("The weights of the model are : ", model.weights)

    #Score
    print("The score is : ","{:.3f}".format(model.score(X_test,y_test)))

    # Draw a roc curve
    best_threhold=draw_roc_curve(model, X_train, y_train)
    print("The score with the best threhold is : ","{:.3f}".format(model.score(X_test,y_test,best_threhold)))










