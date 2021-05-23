from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import pandas as pd
import time


month_to_len = {
    '01':31,
    '02':29,
    '03':31,
    '04':30,
    '05':31,
    '06':30,
    '07':31,
    '08':31,
    '09':30,
    '10':31,
    '11':30,
    '12':31,
}


driver = webdriver.Chrome('/Users/AsgerSturisTang/OneDrive - Danmarks Tekniske Universitet/DTU/6. Semester/Bachelor2021/WebCrawler/chromedriver-1')

url = "https://www.usclimatedata.com/climate/palo-alto/california/united-states/usca0830"
driver.get(url)

timeout = 30
try:
    WebDriverWait(driver, timeout).until(EC.visibility_of_element_located((By.ID, "history-tab")))
except TimeoutException:
    driver.quit()

history = driver.find_element_by_id("history-tab")
history.click()

timeout = 30
try:
    WebDriverWait(driver, timeout).until(EC.visibility_of_element_located((By.ID, "year_month_selector")))
except TimeoutException:
    driver.quit()
time.sleep(1)
datepicker = driver.find_element_by_id("year_month_selector")

for i in range(13):
    datepicker.send_keys(Keys.BACK_SPACE)

output_rows = []

for j in range(2011,2020):
    for i in range(1,13):
        if i < 10:
            month = "0" + str(i)
        else:
            month = str(i)
        year = str(j)
        
        datepicker.send_keys(month + " " + year)
        
        datepicker.send_keys(Keys.ENTER)

        html = driver.find_element_by_class_name("history_table_div").get_attribute("innerHTML")
        soup = BeautifulSoup(html)

        table = soup.find("table")

        for table_row in table.findAll('tr'):
            columns = table_row.findAll('td')
            output_row = []
            for column in columns:
                output_row.append(column.text)
            output_row.append(month)
            output_row.append(year)
            output_rows.append(output_row)
            
        print(month)
        print(year)
        for i in range(month_to_len[month] + 20):
            datepicker.send_keys(Keys.BACK_SPACE)
            
        
df = pd.DataFrame(output_rows)
df.to_csv("weatherdat.csv")