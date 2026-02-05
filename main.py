import pandas as pd 
from src.data_ingestion.ingestor import DataIngestor
from src.data_preparation.cleaner import DataCleaner
from src.data_preprocessing.preprocessing import Preprocessor
from src.model_training.train import Trainer
from src.data_preprocessing.splitter import Splitter

ingestor=DataIngestor("data/heart_attack_dataset.csv")
df=ingestor.load()

cleaner = DataCleaner(df)
df=cleaner.clean()

x_train,y_train,x_test,y_test=Splitter().split(df)

preprocessor= Preprocessor(x_train).build()

trainer=Trainer(preprocessor)
results=trainer.train(x_train,x_test,y_train,y_test)


print(results)