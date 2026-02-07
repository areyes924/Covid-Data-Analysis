import pandas as pd

vac_df = pd.read_csv("C:/Users/noahl/Downloads/Covid_Data_Analysis/Covid-Data-Analysis/data/raw/us_state_vaccinations.csv")

# Ensure date is in datetime format
vac_df["date"] = pd.to_datetime(vac_df["date"])

# Only using totals from mid 2021
vac_df = vac_df[vac_df["date"] == pd.Timestamp("2021-06-01")]

vac_df = vac_df[[
    "date",
    "location",
    "people_vaccinated",
    "people_fully_vaccinated"
]]

vac_df = vac_df.reset_index(drop=True)

valid_locations = [
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut",
    "Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa",
    "Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan",
    "Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada",
    "New Hampshire","New Jersey","New Mexico","New York","North Carolina",
    "North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island",
    "South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont",
    "Virginia","Washington","West Virginia","Wisconsin","Wyoming",
    "District of Columbia","Puerto Rico"
]
vac_df = vac_df[vac_df["location"].isin(valid_locations)]

print(vac_df["location"].value_counts())

print(vac_df.head())

# save cleaned vaccination data
vac_df.to_csv(
    "C:/Users/noahl/Downloads/Covid_Data_Analysis/Covid-Data-Analysis/assembly/cleaned_data/vaccination_2021_06_01.csv",
    index=False
)

    
