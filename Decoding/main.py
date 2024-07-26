#program to auto get the data from the web adress and put into pandas dataframe

import pandas
import numpy
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# Provide the path to your WebDriver executable
service = Service(r'/users/rprocious/downloads/chromedriver_mac64/chromedriver.exe')

# Initialize the WebDriver with the Service object
driver = webdriver.Chrome(service=service)

# Open the URL
driver.get("https://lighthouse.tamucc.edu/qc/008")

# Locate the 'pre' element
elem = driver.find_element(By.TAG_NAME, 'pre')

# Clear the element, send keys, and submit
elem.clear()
elem.send_keys("pycon")
elem.send_keys(Keys.RETURN)

