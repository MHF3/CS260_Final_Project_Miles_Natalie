import csv
import OrderedDict

class Example:

    def __init__(self, features, label):
        """Helper class (like a struct) that stores info about each example."""
        # dictionary. key=feature name: value=feature value for this example
        self.features = features
        self.label = label # in {0, 1, 2, ..., K-1}

class Partition:

    def __init__(self, data, F, K):
        """Store information about a dataset"""
        # list of Examples
        self.data = data
        self.n = len(self.data)

        # dictionary. key=feature name: value=list of possible values
        self.F = F

        # number of classes
        self.K = K
        
def read_csv(filename, switch_points = None):
    
    """
    read the CSV file (from the str filename) into the Partition format.
    filename : name of CSV file to be read into partition format
    return : partition of data
    """
    
    data = []
    F = OrderedDict()
     
    csv_file = csv.reader(open(filename, 'r'), delimiter = ',')
    
    for row in csv_file:
         data.append(row)
    # so now we have a list (data) of rows
    
    # CHANGED
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
    
    if switch_points is None :
        switch_points = {}
        
    for feature in F:

        # if continuous, convert feature (NOTE: modifies data and F_disc)
        if F[feature] == "cont":
            if feature not in switch_points :
                switch_points[feature] = mapping(feature, examples, F_disc)
                
            appply_mapping(feature, examples, F_disc, switch_points[feature])
                

        # if not continuous, just copy over
        else:
            F_disc[feature] = F[feature]

    partition = Partition(examples, F_disc, K)
    
    return partition, switch_points


def mapping(feature, data, F_disc):
    """
    Convert one feature (name f) from continuous to discrete.
    Credit: based on original code from Ameet Soni.
    Split convert one in half into where it makes the mapping and where
        it actually applies it because need to use same mapping for train
        and test
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


def appply_mapping(feature, data, F_disc, switch_points) :

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