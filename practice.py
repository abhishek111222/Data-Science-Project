import requests
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import sqlite3
import os


OWNER = "CSSEGISandData"
REPO = "COVID-19"
PATH = "csse_covid_19_data/csse_covid_19_daily_reports"
URL = f'https://api.github.com/repos/{OWNER}/{REPO}/contents/{PATH}'

     
download_url = []
response = requests.get(URL)


for data in tqdm(response.json()):
    if data["name"].endswith(".csv"):
        download_url.append(data["download_url"])


relabel = {
    "Country/Region" : "Country_Region",
    "Lat" : "Latitude",
    "Long_" : "Longitude",
    'Province/State': 'Province_State'
}




def factor_dataframe(dat, filename):

    for label in dat:
        if label in relabel:
            dat = dat.rename(columns = {label : relabel[label]})

    
    labels = ['Province_State', 'Country_Region', 'Last_Update', 'Confirmed', 'Deaths', 'Recovered']

    if "Last_Update" not in dat:
        dat["Last_Update"] = pd.to_datetime(filename)

    
    for label in labels:
        if label not in dat:
            dat[label] = np.nan

    return dat[labels]


def upload_to_sql(filenames, db_name, debug = False):

    conn = sqlite3.connect(f"{db_name}.db")

    if debug:
        print("Uploading into database")

    for i, file_path in tqdm(list(enumerate(filenames))):

        dat = pd.read_csv(file_path)

        filename = os.path.basename(file_path).split(".")[0]
        print(f"The filename is {filename}")
        dat = factor_dataframe(dat, filename)


        if i == 0:
            dat.to_sql(db_name, con = conn, index = False, if_exists = "replace")
        else:
            dat.to_sql(db_name, con = conn, index = False, if_exists = "append")


upload_to_sql(download_url, "example", debug = True)