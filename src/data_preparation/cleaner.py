import pandas as pd

class DataCleaner():
    def __init__(self,df):
        self.df=df

    def clean(self):
        df = self.df.drop_duplicates()
        return df 