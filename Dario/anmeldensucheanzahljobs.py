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

print("Type in keyword ...")

#es folgt: die Anmeldefunktion (muss man evt weglassen, geht auch ohne)

element = driver.find_element_by_xpath("""//*[@id="login-email"]""")
element.send_keys(str("dario.jussel@bluewin.ch"))
element = driver.find_element_by_xpath("""//*[@id="login-password"]""")
element.send_keys(str("testfuric12"))
element.send_keys(Keys.ENTER)
time.sleep(4)
element = driver.find_element_by_xpath("""//*[@id="jobs-tab-icon"]""").click()
time.sleep(4)
# es folgt: die Suchfunktion

#ab hier problem!!

element = driver.find_element_by_xpath("""//*[@id="ember1181"]/button""").click()
time.sleep(4)
element = driver.find_element_by_xpath("""//*[@id="jobs-search-box-keyword-id-ember1226"]""")
element.send_keys(str("ABB"))
# element = driver.find_element_by_xpath("""//*[@id="location-box"]/button/li-icon""").click()
element = driver.find_element_by_xpath("""//*[@id="jobs-search-box-location-id-ember1226"]""")
element.send_keys(str("Schweiz"))
element.send_keys(Keys.ENTER)

# es folgt: die Textausgabe

print("ABB offers about ")
time.sleep(4)

element = driver.find_element_by_xpath("""/html/body/main/div[2]/div/div[1]/p/span""")
print(element.text)
print("on 26 March")
