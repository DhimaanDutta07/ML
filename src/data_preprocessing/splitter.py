import pandas as pd
from sklearn.model_selection import train_test_split

class Splitter():
    def split(self,df):
        x=df.drop("target",axis=1)
        y=df["target"]
        
        return train_test_split(x,y,test_size=0.2,
            random_state=42)

