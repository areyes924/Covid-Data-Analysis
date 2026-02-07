import pandas as pd

vote_df = pd.read_csv("data/raw/voting.csv")

vote_df = vote_df.drop(columns = ["state_abr", "trump_vote", "biden_vote", "trump_win", "biden_win"])

print(vote_df.head())

vote_df.to_csv(
    "data/processed/clean_vote_df.csv",
    index=False
)