import numpy as np
import pandas as pd
import re
import ast
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import tensorflow_hub as hub



def combineDataSets(file_dir, kaggle_path):
    food_data = pd.DataFrame(columns=['Dish', 'Recipie', 'Ingredients', 'Cleaned Ingredients', 'Image Links'])
    kaggle_data = pd.read_csv(kaggle_path)
    kaggle_data = kaggle_data.rename(columns={"Title": "Dish",
                                              "Instructions": "Recipie",
                                              "Cleaned_Ingredients": "Cleaned Ingredients",
                                              "Image_Name": "Image Links"})
    for i in range(3):
        fileName = file_dir + "/food_data_cleaned" + str(i) + ".csv"
        food_data_web = pd.read_csv(fileName)
        food_data = pd.concat([food_data, food_data_web], ignore_index=True)
    food_data = pd.concat([food_data, kaggle_data], ignore_index=True)
    return food_data


def cleanIngredients(food_recipies):
    lemmetizer = WordNetLemmatizer()
    model_dir = "models"
    model = hub.load(model_dir)
    alt_dict = {'breast': 'chicken'}
    lower = lambda x: x.lower()
    measurement = pd.read_csv('measurements.csv')['List of units'].apply(lower)
    measurement = set(measurement)
    unclean_ingredients = food_recipies['Cleaned Ingredients'].tolist()
    outer_list = []
    embedding_list = []
    for ingredients in unclean_ingredients:
        ingredients = ast.literal_eval(ingredients)
        clean_ingredients = []
        for i in ingredients:
            i = i.lower()
            i = cleanUp(i)
            i = removePunctuation(i)
            i = removeDigits(i)
            i = removeStopWords(i)
            i = makeSingular(i, lemmetizer=lemmetizer)
            i = removeDescriptiveWords(i, alt_dict=alt_dict, measurement=measurement)
            if i is not None:
                clean_ingredients.append(i)
        clean_ingredients = sorted(clean_ingredients)
        combined = "".join(j + " " for j in clean_ingredients)
        combined = combined.rstrip()
        embedding = vectorize([combined], model)
        outer_list.append([combined])
        embedding_list.append(embedding)
    food_recipies['Ingredients List'] = outer_list
    food_recipies["Embeddings"] = embedding_list
    return food_recipies

# Helper methods
def cleanUp(i):
    if (i.find(', ') != -1):
        i = i[:i.find(',')]
    if (i.find('(') != -1):
        i = i[:i.find('(')]
    return i

def removeDigits(i):
    i = re.sub('[0-9]+|/', "", i)
    return i

def removeStopWords(s):
    stop_words = set(stopwords.words('english'))
    clean_str = ""
    for word in s.split():
        if word not in stop_words:
            clean_str = clean_str + word + " "
    return clean_str

def removePunctuation(i):
    i = i.translate(str.maketrans('', '', string.punctuation))
    return i

def makeSingular(i, lemmetizer):
    singular_ingredients = ''
    for word in i.split():
        singular_word = lemmetizer.lemmatize(word)
        singular_ingredients = singular_ingredients + singular_word + " "
    singular_ingredients = singular_ingredients.rstrip()
    return singular_ingredients

def removeNonIngredientWords(i):
    filtered_ingredients = ""
    tokens = word_tokenize(i)
    tags = nltk.pos_tag(tokens)
    for tag in tags:
        if tag[1] == 'NN':
            filtered_ingredients = filtered_ingredients + tag[0] + " "
    if filtered_ingredients != "":
        filtered_ingredients = filtered_ingredients.rstrip()
    return filtered_ingredients

def removeDescriptiveWords(i, alt_dict, measurement):
    clean_i = ""
    for j in alt_dict:
        if i.find(j) != -1:
            i = alt_dict[j]
    i = ''.join(j + " " for j in i.split())
    i = i.rstrip()
    for word in i.split():
        if isWord(word) and word not in measurement:
            clean_i = clean_i + word + " "
    clean_i = clean_i.rstrip()
    if len(clean_i) != 0:
        return clean_i

def isWord(s):
    return s.isalpha()

def vectorize(s, model):
    return np.array((model(s)))

# main method ot execute the code.
def main():
    food_recipies = combineDataSets(file_dir="DataSets", kaggle_path='DataSets/Food Ingredients and Recipe Dataset with Image Name Mapping.csv')
    food_recipies = cleanIngredients(food_recipies)
    food_recipies.to_csv('DataSets/Clean and Clustered Data/food_data_embeddings.csv', index=False)

if __name__ == '__main__':
    main()

