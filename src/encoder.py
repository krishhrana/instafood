import ast

import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np

model_dir = "models"
model = hub.load("models")
def getEmbeddings(s):
    return np.array((model(s)))
food_data = pd.read_csv('DataSets/food_data_combined_cuisines.csv')
ingredients_list = food_data['Ingredients List'].tolist()
embedding_list = []
for i in ingredients_list:
    i = ast.literal_eval(i)
    embedding = getEmbeddings(i)
    embedding_list.append(embedding)
food_data["Embeddings"] = embedding_list
food_data.to_csv('DataSets/merged/food_cuisines_embeddings.csv', index=False)
