import kagglehub
from kagglehub import KaggleDatasetAdapter
from collections import OrderedDict

import file_setup

def main():

    file_setup.load_kaggle_data()
    
    train_partition, switch_points = file_setup.read_csv("data/water_potability_train.csv")
    test_partition, _ = file_setup.read_csv("data/water_potability_test.csv", switch_points) # ignoring switch points return
    
    

if __name__ == '__main__': main()