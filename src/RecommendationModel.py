import pickle
import joblib
from joblib import dump, load
from gensim.models import Word2Vec
import ast
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import tensorflow_hub as hub


class RecommendationModel():
    def __init__(self, data_path, encoder_path):
        self.data = pd.read_csv(data_path)
        self.cluster_model = None
        self.encoder = hub.load(encoder_path)

    def build(self, n_clusters):
        self.cluster_model = KMeans(n_clusters=n_clusters)

    def train(self):
        embeddings = self.data['Embeddings']
        X = [np.fromstring(i[2:-2], dtype='float64', sep=' ') for i in embeddings]
        if self.cluster_model is None:
            return ValueError("Model is Null")
        self.cluster_model.fit(X)
        cluster_id = self.cluster_model.labels_
        self.data['Cluster ID'] = cluster_id
        self.data['Dish ID'] = self.data.index.tolist()

    def save(self, path):
        dump(self.cluster_model, path)
        self.data.to_csv('DataSets/Clean and Clustered Data/food_data_clustered.csv', index=False)

    def load(self, path):
        self.cluster_model = load(path)
        self.data = pd.read_csv('src/DataSets/Clean and Clustered Data/food_data_clustered.csv')

    def predict(self, user_input):
        user_input = self.__processInput(user_input)
        input_embeddings = np.array(self.encoder([user_input])).astype('float64')
        cluster = self.cluster_model.predict(input_embeddings)
        return cluster, input_embeddings

    def recommend(self, user_iput):
        cluster, input_embeddings = self.predict(user_iput)
        return self.__sortByL2(cluster_id=cluster, input_embeddings=input_embeddings)

    def __processInput(self, user_input):
        user_input = user_input.split()
        input_ingredients = sorted(user_input)
        input_ingredients = "".join(j + " " for j in input_ingredients)
        input_ingredients = input_ingredients.rstrip()
        return input_ingredients

    def __sortByL2(self, cluster_id, input_embeddings):
        l2_dict = dict()
        recommended = self.data[self.data['Cluster ID'] == cluster_id[0]]
        recommended_embeddings = recommended['Embeddings'].tolist()
        recommended_embeddings = [np.fromstring(i[2:-2], sep=' ') for i in recommended_embeddings]
        dish_id = recommended['Dish ID'].tolist()
        for i, id in zip(recommended_embeddings, dish_id):
            l2 = np.linalg.norm(i - input_embeddings)
            l2_dict[id] = l2
        l2_dict = sorted(l2_dict.items(), key=lambda item: item[1])
        sorted_recommended = pd.DataFrame(columns=recommended.columns.tolist())
        for i in l2_dict:
            row = recommended[recommended['Dish ID'] == i[0]]
            sorted_recommended = pd.concat([sorted_recommended, row], ignore_index=True)
        return sorted_recommended
