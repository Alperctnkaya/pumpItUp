import sys
import pandas as pd
import numpy as np
import math
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def dropFeatures(features):
    features.remove("id")
    features.remove("wpt_name")
    features.remove("region_code")
    features.remove("scheme_name")
    features.remove("ward")
    features.remove("recorded_by")
    features.remove("extraction_type_group")
    features.remove("payment")
    features.remove("waterpoint_type_group")
    features.remove("date_recorded")
    features.remove("year")
    features.remove("basin")
    features.remove("subvillage")
    features.remove("amount_tsh")


def preprocess(dataset):
    dataset.loc[dataset.construction_year <= 0, dataset.columns == "construction_year"] = 1960
    dataset.amount_tsh = np.log(dataset.amount_tsh + 1)
    dataset.population = np.log(dataset.population+1)

    dataset.insert(1,"year",dataset.date_recorded.str.split("-").str[0].astype(int))
    dataset.insert(2, "month", dataset.date_recorded.str.split("-").str[1].astype(int))
    imputeFeatures(dataset)
    ageOfWaterPoint(dataset)
    groupUniqueValues(dataset)


def groupUniqueValues(dataset):
    factorschange = [x for x in dataset.columns if
                     x not in ['id', 'latitude', 'longitude', 'gps_height', "population",'date_recorded', 'construction_year',
                               'month',"subvillage_name"]]

    for factor in factorschange:
        values_factor = dataset[factor].value_counts()
        lessthen = values_factor[values_factor < 20]
        listnow = dataset.installer.isin(list(lessthen.keys()))
        dataset.loc[listnow, factor] = 'Others'

    for factor in ["gps_height", "population"]:
        values_factor = dataset[factor].value_counts()
        lessthen = values_factor[values_factor < 5]
        listnow = dataset.installer.isin(list(lessthen.keys()))
        dataset.loc[listnow, factor] = 15

def imputeFeatures(dataset):
    imputePopulation(dataset)
    imputeConstructionYear(dataset)
    imputeGpsHeight(dataset)
    imputeLongLat(dataset)
    imputeAmountTsh(dataset)

def imputeGpsHeight(dataset):
    a=dataset[dataset["gps_height"]<5]
    a.iloc[:, dataset.columns == "gps_height"] = np.nan
    dataset[dataset["gps_height"] < 5] = a
    dataset["gps_height"] = dataset.groupby("region").transform(lambda x: x.fillna(x.mean())).gps_height
    dataset["gps_height"] = dataset.groupby("basin").transform(lambda x: x.fillna(x.mean())).gps_height

def imputeLongLat(dataset):
    a=dataset[dataset["longitude"]<1]
    a.iloc[:, dataset.columns == "latitude"] = np.nan
    a.iloc[:, dataset.columns == "longitude"] = np.nan
    dataset[dataset["longitude"] < 1] = a
    dataset["longitude"] = dataset.groupby("region").transform(lambda x: x.fillna(x.mean())).longitude
    dataset["latitude"] = dataset.groupby("region").transform(lambda x: x.fillna(x.mean())).latitude

def imputeAmountTsh(dataset):
    a=dataset[dataset["amount_tsh"]<1]
    a.iloc[:, dataset.columns == "amount_tsh"] = np.nan
    dataset[dataset["amount_tsh"] < 1] = a
    dataset["amount_tsh"] = dataset.groupby(["month","region"]).transform(lambda x: x.fillna(x.mean())).longitude
    dataset["amount_tsh"] = dataset.groupby("region").transform(lambda x: x.fillna(x.mean())).longitude
    dataset["amount_tsh"] = dataset.groupby("basin").transform(lambda x: x.fillna(x.mean())).longitude

def imputePopulation(dataset):
    a=dataset[dataset["population"]<5]
    a.iloc[:, dataset.columns == "population"] = np.nan
    dataset[dataset["population"] < 5] = a
    dataset["population"] = dataset["population"].fillna(dataset.groupby("region")["population"].transform("mean"))
    dataset["population"] = dataset["population"].fillna(dataset.groupby("basin")["population"].transform("mean"))

def imputeConstructionYear(dataset):
    dataset["construction_year"].replace(0,np.NaN, inplace=True)
    dataset["construction_year"] = dataset["construction_year"].fillna(dataset.groupby("installer")["construction_year"].transform("mean"))
    dataset["construction_year"] = dataset["construction_year"].fillna(dataset.groupby("basin")["construction_year"].transform("mean"))
    dataset.construction_year = dataset.construction_year.astype(int)

def scaleColumns(dataset):
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)

def replaceNaNValues(dataset, categoricalFeatures):
    for columnn in categoricalFeatures:
        if columnn not in dataset.columns:
            continue
        dataset[columnn] = dataset[columnn].fillna('n/a')

def ageOfWaterPoint(dataset):
    dataset.insert(1,"age", dataset.year - dataset.construction_year)


def labelEncode(dataset, categoricalFeatures):
    for column in categoricalFeatures:
        if column not in dataset.columns:
            continue
        dataset[column]= dataset[column].astype(str)
        le = LabelEncoder()
        le.fit(dataset[column])
        dataset[column] = le.transform(dataset[column])

def infoGain(dataset, feature, target):
    ent = entropy(dataset, target)
    sum=0
    for i in dataset[feature].unique():
        sv_over_s = dataset[feature].value_counts().to_frame()[feature][i]/len(dataset)
        sum+=sv_over_s*entropy(dataset.loc[dataset[feature] == i], target)
    return ent - sum

def entropy(dataset, target):
    entropy = 0

    for i in dataset[target].unique():
        pi=dataset[target].value_counts().to_frame()[target][i]/len(dataset)
        entropy += -pi*math.log(pi,2)

    return entropy

def dropUnknowns(trainSet):
    trainSet.drop(trainSet[ (trainSet["management"] == "unknown") & (trainSet["payment_type"] == "unknown")].index, inplace=True)
    trainSet.drop(trainSet[trainSet["water_quality"] == "unknown"].index, inplace=True)
    trainSet.drop(trainSet[trainSet["quantity"] == "unknown"].index, inplace = True)