import time
import pandas as pd
import re
import json
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

import requests
import urllib.request

main_url = "https://ch.linkedin.com/jobs"
source = "Linkedin.com"

driver = webdriver.Safari()
driver.maximize_window()

driver.get(main_url)
time.sleep(4)



#folgender Code ist für das Anmelden bei Linkedin --> für Benutzerspezifische Suche
'''
element = driver.find_element_by_xpath("""//*[@id="login-email"]""")
element.send_keys(str("<ENTER USERS EMAIL>"))
element = driver.find_element_by_xpath("""//*[@id="login-password"]""")
element.send_keys(str("<ENTER USERS PASSWORD"))
element.send_keys(Keys.ENTER)
time.sleep(4)
element = driver.find_element_by_xpath("""//*[@id="jobs-tab-icon"]""").click()
time.sleep(4)
'''
#Loop für ausgewählte Unternehmen oder einzelne Unternehmen eingeben!
'''
list = ["ABB","Apple","Tesla"]
for i in list:
    '''

element = driver.find_element_by_xpath("""//*[@id="keyword-box-input"]""")
element.send_keys(str("<ENTER COMPANY>"))
element = driver.find_element_by_xpath("""//*[@id="location-box-input"]""")
element.send_keys(str("Schweiz"))
element.send_keys(Keys.ENTER)
time.sleep(4)
element = driver.find_element_by_xpath("""/html/body/main/div[2]/div/div[1]/p/span""")
print(element.text)
print("on <ENTER DATE>")
