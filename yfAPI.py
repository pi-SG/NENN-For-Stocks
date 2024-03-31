# python yfAPI.py
import yfinance as yf
import pandas
import time
from datetime import datetime, timedelta

from pyfinviz.screener import Screener

page = 439

init_time = datetime.now()

screener = Screener(pages=[x for x in range(1, page)])

list_ticker = []
for i in range(85, page):
  for j in range(len(screener.data_frames[i])):
    list_ticker.append(screener.data_frames[i].Ticker[j])
    list(list_ticker)

amount = len(list_ticker)
for k in range(amount):
  ticker= yf.Ticker(list_ticker[k])

  hist = ticker.history(start="1900-01-01", end="2024-01-01", interval="1d")

  OHLC = [hist["Open"].tolist(), hist["High"].tolist(), hist["Low"].tolist(), hist["Close"].tolist(),hist["Volume"].tolist()]

  path = "C:\\Users\\grink\\OneDrive\\Documents\\PythonScripts\\DailyStockOHLC\\"+list_ticker[k]+"OHLC.txt"

  with open(path, "a") as f:
    for d in range(len(OHLC[0])):
      f.write(
        str(OHLC[0][d]) + " " + 
        str(OHLC[1][d]) + " " + 
        str(OHLC[2][d]) + " " + 
        str(OHLC[3][d]) + " " + 
        str(OHLC[4][d]) + "\n"
      )
    f.close()

  print("Estimated Time Left: " + 
    str((datetime.now() - init_time) / (k + 1) * amount - (datetime.now() - init_time)))

  time.sleep(1)













