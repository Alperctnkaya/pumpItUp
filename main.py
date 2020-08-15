from preprocess import *
from modelTrain import *
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
if __name__ == "__main__":
    sys.argv = ["labels.csv", "test.csv" , "training.csv"]

    labels = pd.read_csv(sys.argv[0])
    testSet = pd.read_csv(sys.argv[1])
    trainSet = pd.read_csv(sys.argv[2])
    trainSet = pd.merge(trainSet, labels)

    status = ["functional", "non functional" , "functional needs repair"]
    categoricalFeatures=["date_recorded", "funder", "installer","wpt_name", "basin", "subvillage", "region","lga","ward", "public_meeting",
                       "recorded_by", "scheme_management", "scheme_name", "permit", "extraction_type", "extraction_type_group", "extraction_type_class",
                       "management", "management_group", "payment", "payment_type", "water_quality","quality_group", "quantity","quantity_group", "source",
                       "source_type", "source_class", "waterpoint_type", "waterpoint_type_group"]


    preprocess(trainSet)

    q=trainSet

    features = list(trainSet.columns.values[:-1])
    dropFeatures(features)

    labelEncode(trainSet, features)

    X = trainSet.iloc[:,:-1]
    y = trainSet.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(trainSet, labels["status_group"], test_size=0.1,random_state=0)

    randomForestCLF = randomForest(X[features], y, 300)
    accuracyRF = randomForestCLF.score(X_test[features], y_test)

    xgbClassifier = xgboost(X[features], y)
    accuracyXG=xgbClassifier.score(X_test[features], y_test)
    print(accuracyRF)
    print(accuracyXG)

