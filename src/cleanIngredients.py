import numpy as np
import pandas as pd
import re
import ast
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import inflect
from nltk.stem import WordNetLemmatizer
#nltk.download('omw-1.4')
p = inflect.engine()
lemmetizer = WordNetLemmatizer()
alt_dict = {
    'breast': 'chicken'
}
nuisance = {'semisweet', 'die', 'ml', 'wheat', 'slice', 'nutrition', 'broken', 'pan', 'jack', 'tender', 'light', 'cut', 'cube', 'lowfat',
            'ground', 'jar', 'blend', 'bow', 'tie', 'package', 'optional', 'approx', 'taste', 'chopped', 'cherry', 'mix salad leaf',
            'fillet', 'rotini', 'crosswise', 'container', 'box', 'baby', 'g', 'raw', 'sheet', 'choice', 'flake', 'ripe',
            'chorizo', 'salad', 'mix', 'leaf', 'part', 'multigrain', 'split', 'type', 'course', 'plum', 'cold',
            'plain', 'remainder', 'elbow', 'frying', 'water period flavor meld simmer', 'membrane', 'rib', 'hickory smoke salt liquid smoke',
            'mine', 'russet', 'brand', 'favorite'}

lower = lambda x: x.lower()
measurement = pd.read_csv('measurements.csv')['List of units'].apply(lower)
measurement = set(measurement)
print(measurement)
vocabulary = nltk.FreqDist()

def cleanIngredients(unclean_list):
    outer_list = []
    for ingredients in unclean_list:
        ingredients = ast.literal_eval(ingredients)
        clean_ingredients = []
        for i in ingredients:
            i = i.lower()
            i = cleanUp(i)
            i = removePunctuation(i)
            i = removeDigits(i)
            i = removeStopWords(i)
            i = makeSingular(i)
            #i = removeNonIngredientWords(i)
            i = removeDescriptiveWords(i)

            if i is not None:
                clean_ingredients.append(i)
        clean_ingredients = sorted(clean_ingredients)
        combined = "".join(j + " " for j in clean_ingredients)
        combined = combined.rstrip()
        outer_list.append([combined])
    return outer_list


#Helper methods
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

def makeSingular(i):
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

def removeDescriptiveWords(i):
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

food_recipies = pd.read_csv('DataSets/Clean and Clustered Data/food_data_combined.csv')
unclean_ingredients = food_recipies['Cleaned Ingredients'].tolist()
clean = cleanIngredients(unclean_ingredients)
food_recipies['Ingredients List'] = clean
food_recipies.to_csv('DataSets/Clean and Clstered/food_data_combined_cuisines.csv', index = False)
#print(removeDescriptiveWords(ast.literal_eval(unclean_ingredients[0])))
for i in clean:
    vocabulary.update(i)

for ing, freq in vocabulary.most_common(200):
    print(ing, freq)
