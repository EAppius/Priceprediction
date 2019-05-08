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

main_url = "https://www.linkedin.com"
source = "Linkedin.com"


driver = webdriver.chrome()
driver.maximize_window()

driver.get(main_url)
time.sleep(4)

print("Type in keyword ...")

element = driver.find_element_by_xpath("""//*[@id="login-email"]""")
element.send_keys(str("dario.jussel@bluewin.ch"))
element = driver.find_element_by_xpath("""//*[@id="login-password"]""")
element.send_keys(str("testfuric12"))
element.send_keys(Keys.ENTER)
time.sleep(4)
element = driver.find_element_by_xpath("""//*[@id="jobs-tab-icon"]""").click()
time.sleep(4)

#funktioniert bis hier hin


#time.sleep(4)
#element = driver.find_element_by_xpath("""//*[@id="jobs-search-box-keyword-id-ember477"]""")
#element.send_keys(str("ABB"))
#time.sleep(4)
#element.send_keys(Keys.ENTER)
#print("Sent request")

element = driver.find_element_by_xpath("""//*[@id="jobs-search-box-keyword-id-ember1096"]""")
element.send_keys(str("ABB"))
time.sleep(4)
element.send_keys(Keys.ENTER)
time.sleep(4)

#ab hier web web_crawler
element = driver.find_element_by_xpath("""//*[@id="ember5"]/div[7]/div[2]/section[1]/div[4]/div/div/div[1]/div[1]/div[1]/div""")
System.out.println
