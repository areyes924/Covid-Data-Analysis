import pandas as pd
import numpy as np

vote_df = pd.read_csv("data/raw/voting.csv")

vote_df = vote_df.drop(columns = ["state_abr", "trump_vote", "biden_vote", "trump_win", "biden_win"])

print(vote_df.head())

# Puerto Rico does not vote in the general election. Adding NAN values as a result.
pr_row = pd.DataFrame([{
    "state": "Puerto Rico",
    "trump_pct": np.nan,
    "biden_pct": np.nan,
}])

vote_df = pd.concat([vote_df, pr_row], ignore_index=True)

vote_df.to_csv(
    "data/processed/clean_vote_df.csv",
    index=False
)