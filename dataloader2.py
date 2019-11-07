import pandas as pd


class Dataloader2(): 

    def __init__(self, path):
        
        '''
           path -- Path to the Bike Sharing Dataset CSV file.
        '''
        self.path = path
        self.data = pd.read_csv(self.path)
      

    def getData(self):

        #Split data into train, validation and test set with 60:20:20 ratio
        split_train = int(60 / 100 * len(self.data)) 
        split_val = int(80 / 100 * len(self.data)) 
        train = self.data[:split_train]
        val = self.data[split_train:split_val]
        test = self.data[split_train:]
        return train, val, test

    def getAllData(self):
        return self.data

