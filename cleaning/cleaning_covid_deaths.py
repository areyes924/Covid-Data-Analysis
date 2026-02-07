import pandas as pd
import os

# Input/Outputs
cov = pd.read_csv("data/raw/covid_dataset.csv")
pop = pd.read_csv("data/raw/age.csv")
output = "data/processed/covid_stats_per_100k"


cov = cov.drop(columns=["fips"], errors="ignore")
pop = pop.drop(columns=["population_65_plus_thousands","population_65_plus_percent"], errors="ignore")

print(cov.info())
print(pop.info())

# Grabbing last reported date
cov["date"] = pd.to_datetime(cov["date"])
idx = cov.groupby("state")["date"].idxmax()
state_totals = cov.loc[idx, ["state", "cases", "deaths"]]


final = state_totals.merge(pop, on="state", how="left")
final[final["total_population_thousands"].isna()]["state"].unique()

final["cases_per_100k"] = final["cases"] / (final["total_population_thousands"] * 1000) * 100000
final["deaths_per_100k"] = final["deaths"] / (final["total_population_thousands"] * 1000) * 100000
final["death_rate"] = final["deaths"] / final["cases"]

# Final Clean and Filter

territories = [
    "American Samoa",
    "Guam",
    "Northern Mariana Islands",
    "Virgin Islands"
]

final = final[~final["state"].isin(territories)]

# print(final.head())
# print(len(final))

print(final.info())
final = final[["state", "cases_per_100k", "deaths_per_100k", "death_rate"]]

os.makedirs("data/processed", exist_ok=True)
final.to_csv(f"{output}.csv", index=False)