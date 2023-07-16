import pandas as pd
import datetime

from binance.client import Client
from binance import AsyncClient, BinanceSocketManager


async def connect():
    import time

    client = Client()
    bm = BinanceSocketManager(client)
    ts = bm.kline_socket('BTCUSDT')

    # enter the context manager
    await ts.__aenter__()

    # receive a message
    msg = await ts.recv()
    print(msg)

    # exit the context manager
    await ts.__aexit__(None, None, None)


def download(title, start_date, end_date):

    def convert(ms):
        date = datetime.datetime.fromtimestamp(ms / 1000)
        return date.strftime("%b %d %Y %H:%M:%S")

    client = Client()

    req = client.get_historical_klines(title, Client.KLINE_INTERVAL_1MINUTE, start_date, end_date)
    df = pd.DataFrame(req, columns=['dateTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
    df['dateTime'] = df['dateTime'].apply(convert)

    # Keep only specific columns
    columns_to_keep = ['dateTime', 'close', 'volume']
    df = df.loc[:, columns_to_keep]

    # Rename column names
    df = df.rename(columns={'dateTime': 'Date', 'close': 'Price_BTCUSD'})

    df.to_csv("data/binance_data/clean_1MIN.csv", index=False)
