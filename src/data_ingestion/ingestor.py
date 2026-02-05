import pandas as pd

class DataIngestor():
    def __init__(self,path):
        self.path="data/heart_attack_dataset.csv"

    def load(self):
        df=pd.read_csv(self.path)
        return df 