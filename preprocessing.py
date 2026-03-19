import pandas as pd
from sklearn.model_selection import train_test_split
from config import TARGET_COLUMN, TEST_SIZE, RANDOM_SEED

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RoomsPerHousehold"] = df["AveRooms"] / df["AveOccup"]
    df["BedroomsRatio"] = df["AveBedrms"] / df["AveRooms"]
    df["PopulationPerHousehold"] = df["Population"] / df["AveOccup"]
    df["IncomePerHousehold"] = df["MedInc"] / df["AveOccup"]
    return df

def split_features_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN):
    X = df.drop(columns=target_column)
    y = df[target_column]
    return X, y

def train_test_split_data(X, y, test_size: float = TEST_SIZE, random_state: int = RANDOM_SEED):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)