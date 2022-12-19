from selenium import webdriver as wb
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
import numpy as np
import pandas as pd
import re

driver = wb.Chrome()
def getRecipieLinks(landing_link):
    landing = landing_link
    driver.get(landing)
    links = []
    titles = []

    recipie_links = driver.find_elements(By.XPATH, '//h2[@class="title"]/a')
    for i in recipie_links:
        links.append(i.get_attribute('href'))
        titles.append(i.text)
    #print(len(links))
    return links, titles

def getInfo(links, titles):
    dataset = pd.DataFrame(columns=['Dish', 'Recipie', 'Ingredients', 'Cleaned Ingredients', 'Image Links'])
    for l in range(len(links)):
        recipie = ""
        ingredients_list = []
        clean_ingredients = []
        driver.get(links[l])
        #dish = driver.find_element(By.XPATH, '//h1[@class="o-AssetTitle__a-Headline"]/span[@class="o-AssetTitle__a-HeadlineText"]')
        ingredients = driver.find_elements(By.XPATH, '//span[@class="ingredient-text svelte-1apyilh"]')
        ingredients_quantity = driver.find_elements(By.XPATH, '//span[@class="ingredient-quantity svelte-1apyilh"]')
        steps = driver.find_elements(By.XPATH, '//li[@class="direction svelte-1apyilh"]')
        image_src_links = driver.find_element(By.XPATH, '//div[@class="primary-image svelte-tdj2pb"]/img')
        #Images
        img_src_list = image_src_links.get_attribute('srcset').split(', ')
        img_src = img_src_list[0]
        img_src = img_src[:re.search(r' ', img_src).start()]
        #Recipie
        for c, s in enumerate(steps):
            recipie = recipie + " Step " + str(c + 1) + ": " + s.text
        #Ingredients
        for ig, igq in zip(ingredients, ingredients_quantity):
            clean_ingredients.append(ig.text)
            ingredients_list.append(igq.text + " " + ig.text)

        row_dict = {
            'Dish': titles[l],
            'Recipie': recipie,
            'Ingredients': [ingredients_list[1:]],
            'Cleaned Ingredients': [clean_ingredients],
            'Image Links': img_src
        }
        row_df = pd.DataFrame(row_dict)
        dataset = pd.concat([dataset, row_df], ignore_index=True)
        dataset.reset_index()
    return dataset

def getCuisineLinks(landing):
    cuisine_links = []
    driver.get(landing)
    #parent = driver.find_element(By.XPATH, "//li[span/text()='Cuisine']")
    links = driver.find_elements(By.XPATH, "//li[span/text()='Cuisine']/ul/li[@class='dropdown__item svelte-14sej0u']/a")
    for l in links:
        print(l.get_attribute('innerHTML'))
        cuisine_links.append([l.get_attribute('innerHTML').rstrip().lstrip(), l.get_attribute('href')])
    return np.array(cuisine_links)


homepage = "https://www.food.com/"
links = getCuisineLinks(homepage)
for l, link in enumerate(links[:, 1]):
    scraped_links, scraped_titles = getRecipieLinks(link)
    print(scraped_links)
    print(scraped_titles)
    print(len(scraped_links), len(scraped_titles))
    scraped_recipies = getInfo(scraped_links, scraped_titles)
    print(scraped_recipies)
    fileName = "/Users/krishrana/PycharmProjects/InstaFood/v2/src/DataSets/food_data_cleaned" + str(l) + ".csv"
    scraped_recipies.to_csv(fileName, index=False)

