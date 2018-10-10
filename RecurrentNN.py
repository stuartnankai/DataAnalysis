from collections import deque
import pandas as pd
from sklearn import preprocessing
import random
import numpy as np

# df = pd.read_csv("crypto_data/LTC-USD.csv",names=['time', 'low', 'high', 'open', 'close', 'volume'])
#
# print(df.head())


SEQ_LEN = 60  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 3  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "LTC-USD"

main_df = pd.DataFrame()

ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]



def preprocess_df(df):
    df = df.drop("Future",1)

    for col in df.columns:
        if col!= "Target":
            df[col]= df[col].pct_change() # Percentage change between the current and a prior element.
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)

    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)  # shuffle for good measure.

    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if it's a "not buy"
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # otherwise if the target is a 1...
            buys.append([seq, target])  # it's a buy!

    random.shuffle(buys)  # shuffle the buys
    random.shuffle(sells)  # shuffle the sells!

    lower = min(len(buys), len(sells))  # what's the shorter length?

    buys = buys[:lower]  # make sure both lists are only up to the shortest length.
    sells = sells[:lower]  # make sure both lists are only up to the shortest length.

    sequential_data = buys + sells  # add them together
    random.shuffle(
        sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array! ..import numpy as np


def classify(current, future):
    if float(current) > float(future):
        return 0
    else:
        return 1


for ratio in ratios:
    # print(ratio)
    dataset = f'crypto_data/{ratio}.csv'
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)
    df.set_index("time", inplace=True)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df.fillna(method='ffill', inplace=True)
main_df.dropna(inplace=True)
# print(main_df.head())

main_df['Future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['Target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['Future']))
main_df.dropna(inplace=True)
# print(main_df.tail())

times = sorted(main_df.index.values)  # get the times
# print(times)
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]  # get the last 5% of the times
# print(last_5pct)

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")