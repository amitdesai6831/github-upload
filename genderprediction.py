import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score, f1_score, plot_confusion_matrix, precision_score, recall_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import itertools

df=pd.read_csv('data_model.csv')

df.head()

df.duplicated().sum()

df.shape

df=df.drop_duplicates()

df.shape

df.gender.value_counts()
dum_var = {'m': 1,'f': 0}

df['gender']=[dum_var[items] for items in df['gender']]

df.head()

df.name.values

cv = CountVectorizer()
X = cv.fit_transform(df.name.values.astype('U'))

x_array=X.toarray()

X=pd.DataFrame(x_array)

X.head()

y=df.gender

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = MultinomialNB()
model.fit(x_train,y_train)

model.score(x_test,y_test)

prediction = model.predict(x_test)

pred_probs_test = model.predict_proba(x_test)

# Method to show different evaluation metrics  
def show_model_metrics(y_test,y_pred,model_name):
    cp = confusion_matrix(y_test,y_pred)
    plt.figure()
    draw_confusion_matrix(cp)
    plt.show()
    
    accuracy = round(accuracy_score(y_test,y_pred),2)
    recall = round(recall_score(y_test,y_pred),2)
    precision = round(precision_score(y_test,y_pred),2)
    auc = round(roc_auc_score(y_test,y_pred),2)
    f1 = round(f1_score(y_test,y_pred),2)
    
    data = [[model_name,accuracy,recall,precision,auc,f1]] 
    df1 = pd.DataFrame(data, columns = ['Model', 'Accuracy','Precision','Recall','AUC','F1'])
    return df1 
    
    # Plot ROC AUC Curve
def roc_auc_curve(X_ts, y_ts, y_pred_probability, classifier_name):
    y_pred_prob = y_pred_probability[:,1]
    fpr, tpr, thresholds = roc_curve(y_ts, y_pred_prob)
    plt.plot([0,1],[0,1], 'k--')
    plt.plot(fpr, tpr, label=f'{classifier_name}')
    plt.xlabel('Flase +ve rate')
    plt.ylabel('True +ve rate')
    plt.title(f'{classifier_name} - ROC Curve')
    plt.show()
    
    return print(f'AUC score (ROC): {roc_auc_score(y_ts, y_pred_prob)}\n')
    
    
    
    # Method to plot confusion matrix
def draw_confusion_matrix(cm):
    classes=[0,1]
    cmap=plt.cm.BuPu
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    
    show_model_metrics(y_test, prediction, "MultinomialNB")
    
    roc_auc_curve(x_test, y_test, pred_probs_test, 'MultinomialNB')
