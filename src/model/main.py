import os
import glob
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import mlflow


# define functions
def main(args):
    mlflow.autolog()  # enable auto logging

    # read in data
    df = get_csvs_df(args.training_data)

    # Cleanse data
    final_df = cleanse_data(df)

    # split data
    X_train, X_test, y_train, y_test = split_data(final_df)

    # train model
    train_model(X_train, X_test, y_train, y_test)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def cleanse_data(df):
    taxi_df = df[['vendorID', 'passengerCount', 'tripDistance',
                  'pickupLongitude', 'pickupLatitude',
                  'dropoffLongitude', 'dropoffLatitude', 'totalAmount']]
    final_df = taxi_df.query("pickupLatitude>=40.53 and pickupLatitude<=40.88")
    final_df = final_df.query(
        "pickupLongitude>=-74.09 and pickupLongitude<=-73.72")
    final_df = final_df.query("tripDistance>=0.25 and tripDistance<31")
    final_df = final_df.query("passengerCount>0 and totalAmount>0")
    columns_to_remove_for_training = ["pickupLongitude",
                                      "pickupLatitude",
                                      "dropoffLongitude",
                                      "dropoffLatitude"]
    for col in columns_to_remove_for_training:
        final_df.pop(col)
    return final_df


def split_data(final_df):
    X = final_df.drop(["totalAmount"], axis=1)
    y = final_df["totalAmount"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=223)
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
