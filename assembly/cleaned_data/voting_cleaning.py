import pandas as pd

vote_df = pd.read_csv("C:/Users/noahl/Downloads/Covid_Data_Analysis/Covid-Data-Analysis/data/raw/voting.csv")

vote_df = vote_df.drop(columns = ["state_abr", "trump_vote", "biden_vote", "trump_win", "biden_win"])

print(vote_df.head())

vote_df.to_csv(
    "C:/Users/noahl/Downloads/Covid_Data_Analysis/Covid-Data-Analysis/assembly/cleaned_data/clean_vote_df.csv",
    index=False
)