from sklearn.model_selection import train_test_split
import  pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def splitDataset(dataset):
    X = dataset.iloc[:,:-1]
    y = dataset.iloc[:, -1]
    return train_test_split(X,y, test_size=0.1, random_state=0)

def plot_features_importances(classifier, X_train):
    series = pd.Series(classifier.feature_importances_, index=X_train.columns).sort_values(ascending=True)
    series = series.plot(kind='barh', figsize=(10,10))
    plt.xlabel('Feature importance')
    plt.ylabel('Features')
    plt.show()

def randomForest(x, y, estimator):
    #max features dene
    clf = RandomForestClassifier(max_depth=22, n_estimators=estimator, criterion= "entropy")
    clf.fit(x, y)
    return clf

def xgboost(x,y):
    clf = xgb.XGBClassifier(nthread=4, num_class=3,
                        min_child_weight=3, max_depth=22,
                        gamma=0.5, scale_pos_weight=0.8,
                        subsample=0.75, colsample_bytree = 0.8,
                        objective='multi:softmax')
    clf.fit(x,y)
    return clf

def result(clf, x_test, y_test):
    accuracy = clf.score(x_test, y_test)
    return accuracy

def score(prediction, y):
    true=(y == prediction).value_counts().iloc[0]
    return true/len(prediction)