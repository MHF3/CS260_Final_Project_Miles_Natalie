"""
Contains helper functions ran in main that read data, create train and test partitions, then convert continuous features into discrete
Author : Natalie Queenan
Date : 
"""

import csv
import kagglehub

from collections import OrderedDict
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split

###########################################################################################################
# CLASSES
###########################################################################################################


class Example:

    def __init__(self, features : dict[str, str], label : str):
        """Helper class (like a struct) that stores info about each example."""
        # dictionary. key=feature name: value=feature value for this example
        self.features : dict[str, str] = features
        self.label : str = label # in {0, 1, 2, ..., K-1}


class Partition:

    def __init__(self, data : list[Example], F : dict[str, list[str]], K : int):
        """Store information about a dataset"""
        # list of Examples
        self.data : list[Example] = data
        self.n : int = len(self.data)

        # dictionary. key=feature name: value=list of possible values
        self.F : dict[str, list[str]] = F

        # number of classes
        self.K : int = K
   
   
###########################################################################################################
# DATA LOADING / PUTTING INTO FORMAT WE NEED FOR REST OF PROJECT
###########################################################################################################
        
        
def load_kaggle_data() :
    """
    Got code in beginning of this method from kaggle on the dataset we choose
    Loads kaggle dataset, drops rows that have null values, and uses sklearn to split it into train (.8) and test (.2). 
        Converts from df to csv to use read_csv() method and puts them in data folder
    """
    
    # Set the path to the file you'd like to load
    file_path : str = "water_potability.csv"

    # Load the latest version
    df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "devanshibavaria/water-potability-dataset-with-10-parameteres",
    file_path,
    # Provide any additional arguments like
    # sql_query or pandas_kwargs. See the
    # documenation for more information:
    # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )
    
    df = df.dropna() # removing rows with null values (can change later if that's not how we want to deal with it)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42) # random_state = 42 makes it the same split every time
    
    # save as csv so we can use read_csv() method
    df.to_csv("data/water_potability.csv", index=False)
    train_df.to_csv("data/water_potability_train.csv", index=False)
    test_df.to_csv("data/water_potability_test.csv", index=False)

        
def read_csv(filename : str , switch_points : list[float] = None):
    """
    Read the CSV file (from the str filename) into the Partition format.
    Uses mapping() and apply_mapping() to convert continuous features into discrete features
    Most of this code is recycled from earlier labs
    filename : name of CSV file to be read into partition format
    return : partition of data
    """
    
    data  = []
    F = OrderedDict()
     
    csv_file = csv.reader(open(filename, 'r'), delimiter = ',')
    
    for row in csv_file:
         data.append(row)
    # so now we have a list (data) of rows
    
    # feature_names are from the headers
    feature_names = data[0][:-1] 
     
    # creating F
    for feature in feature_names :
        F[feature] = "cont"
            
    # getting K
    class_values = []
    for row in data[1:] :
        # getting the value of class
        value = row[-1]
        if value not in class_values :
            class_values.append(value)
    K = len(class_values)

    # making example objects
    examples = []
    for row in data[1:] :
        X_dict = {}
        for i, feature in enumerate(feature_names) :
            X_dict[feature] = float(row[i]) # changed from str to float
        label = row[-1]
        examples.append(Example(X_dict, label))

    # convert continuous features to discrete
    F_disc = {}
    
    # should create dict for train data but then pass that dict in when used on test data so 
    # both partitions have the same discrete features
    if switch_points is None :
        switch_points = {}
        
    for feature in F:

        # if continuous, convert feature (NOTE: modifies data and F_disc)
        if F[feature] == "cont":
            if feature not in switch_points :
                switch_points[feature] = mapping(feature, examples)
                
            apply_mapping(feature, examples, F_disc, switch_points[feature])

        # if not continuous, just copy over
        else:
            F_disc[feature] = F[feature]

    partition = Partition(examples, F_disc, K)
    
    return partition, switch_points


def mapping(feature, data):
    """
    Creates the label changes to convert one feature from continuous to discrete
    Credit: based on original code from Ameet Soni.
    feature : cont feature to convert to discrete
    data : data where feature came from
    return : switch_points -- discrete label changes for the continuous feature 
    """

    # first combine the feature values (for f) and the labels
    combineXy = []
    for example in data:
        combineXy.append([example.features[feature],example.label])
    combineXy.sort(key=lambda elem: elem[0]) # sort by feature

    # first need to merge uniques
    unique = []
    u_label = {}
    for elem in combineXy:
        if elem[0] not in unique:
            unique.append(elem[0])
            u_label[elem[0]] = elem[1]
        else:
            if u_label[elem[0]] != elem[1]:
                u_label[elem[0]] = None

    # find switch points (label changes)
    switch_points = []
    for j in range(len(unique)-1):
        if u_label[unique[j]] != u_label[unique[j+1]] or u_label[unique[j]] \
            == None:
            switch_points.append((unique[j]+unique[j+1])/2) # midpoint

    return switch_points


def apply_mapping(feature, data, F_disc, switch_points) :
    """
    For each new discrete label created for the continuous feature, creates a new binary feature that says if the original 
        feature value is less than or equal to that point, then removes the original feature
    Credit: based on original code from Ameet Soni.
    feature : cont feature to convert to discrete
    data : data where feature came from
    F_disc : possible values for feature
    switch_points : discrete label changes for the continuous feature 
    """
    
    # add a feature for each switch point (keep feature vals as strings)
    for s in switch_points:
        key = feature+"<="+str(s)
        for i in range(len(data)):
            if data[i].features[feature] <= s:
                data[i].features[key] = "True"
            else:
                data[i].features[key] = "False"
        F_disc[key] = ["False", "True"]

    # delete this feature from all the examples
    for example in data:
        del example.features[feature]