import pandas as pd

# Load Datasets

cov = pd.read_csv("data/processed/covid_stats_per_100k.csv")
vacc = pd.read_csv("data/processed/vaccination_summary.csv")
votes = pd.read_csv("data/processed/clean_vote_df.csv")
policy = pd.read_csv("data/processed/clean_policy.csv")
age = pd.read_csv("data/raw/age.csv")
edu = pd.read_csv("data/raw/education.csv")
income = pd.read_csv("data/raw/income.csv")
race = pd.read_csv("data/raw/race.csv")

# Output
output = "master_dataset/master_dataset"

# Merge

final = cov \
    .merge(vacc,   on="state", how="left") \
    .merge(votes,  on="state", how="left") \
    .merge(policy, on="state", how="left") \
    .merge(age,    on="state", how="left") \
    .merge(edu,    on="state", how="left") \
    .merge(income, on="state", how="left") \
    .merge(race,   on="state", how="left")

final[final["total_population_thousands"].isna()]["state"].unique()

redundant = [
    "pct_full_2021-09-01",
    "pct_full_2022-09-01",
    "total_population_thousands",
    "population_65_plus_thousands",
    "population_over_25",
    "population_with_hs",
    "population_with_bachelors",
    "population_with_advanced"
]

final = final.drop(columns=redundant)

print(final.info())

final.to_csv(f"{output}.csv", index=False)