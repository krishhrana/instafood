import os.path
import time

from gensim.models import Word2Vec
import ast
import pandas as pd
import numpy as np
from joblib import load
import tensorflow_hub as hub
import streamlit as st
import zipfile

from RecommendationModel import RecommendationModel

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 10000000)



def main():
    ingredients = st.text_area("What ingredients do you have?")
    if (ingredients != ""):
        with st.spinner('Asking the Chef...'):
            time.sleep(5)
        st.success('Done!')
        recommendations = model.recommend(ingredients)
        title = recommendations['Dish'].tolist()
        direction = recommendations['Recipie'].tolist()
        ingredients = recommendations['Ingredients'].tolist()
        image_links = recommendations['Image Links'].tolist()

        for i in range(len(title)):
            st.header(str(i + 1) + ". " + title[i])
            if (image_links[i].startswith("http")):
                st.image(image_links[i])
            else:
                st.image("DataSets/Food Images/Food Images/" + image_links[i] + ".jpg")
            col1, col2 = st.columns(2, gap="medium")
            col1.header("Directions")
            col2.header("Ingredients")
            for j in str(direction[i]).split('. '):
                col1.markdown("- " + j.lstrip().rstrip())
            for j in ast.literal_eval(ingredients[i]):
                col2.markdown("- " + j.lstrip().rstrip())
            c = st.container()




# onion garlic hummus egg bread tortilla

if __name__ == '__main__':
    if not os.path.exists("src/DataSets"):
        with zipfile.ZipFile("src/DataSets.zip", 'r') as zip_ref:
            zip_ref.extractall()
            time.sleep(10)
    if not os.path.exists("models"):
        with zipfile.ZipFile("src/models.zip", 'r') as zip_ref:
            zip_ref.extractall()
            time.sleep(10)
    model = RecommendationModel(data_path='DataSets/Clean and Clustered Data/food_data_clustered.csv',
                                encoder_path='models')
    model.load(path='models/v2/kmeans_500.joblib')
    st.title('InstaFood')
    main()
