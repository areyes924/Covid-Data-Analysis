import pandas as pd
import numpy as np

policy_df = pd.read_csv("data/raw/COVID-19 US state policy.csv")


# initial clean
cols_to_keep = [
    "STATE",
    "STAYHOME",
    "END_STHM",
    "CLBSNS",
    "END_BSNS",
    "FM_ALL",
    "FMNOENF",
    "AGEPRIORITY"
]

# keep only those columns
policy_df = policy_df[cols_to_keep]

# reset index (optional but clean)
policy_df = policy_df.reset_index(drop=True)

s = policy_df["STAYHOME"].astype(str).str.strip()

policy_df = policy_df[s.str.match(r"^(0|[0-9]{1,2}/[0-9]{1,2}/[0-9]{4})$")].reset_index(drop=True)

import numpy as np
import pandas as pd

# this builts the features we need
# goes from dates to numerical day values
def build_policy_features(df, start_col, end_col, prefix):
    # parse dates
    start_dt = pd.to_datetime(df[start_col], format="%m/%d/%Y", errors="coerce")
    end_dt = pd.to_datetime(df[end_col], format="%m/%d/%Y", errors="coerce")

    # earliest adoption
    earliest = start_dt.min()

    # timing: days since earliest (0 if never adopted)
    df[f"{prefix}_days_since_earliest_adoption"] = (
        (start_dt - earliest).dt.days.fillna(-1)
    )

    # duration: end - start (0 if never adopted)
    df[f"{prefix}_duration_days"] = np.where(
        start_dt.notna() & end_dt.notna(),
        (end_dt - start_dt).dt.days,
        0
    )

    return df

# use function for stay home mandates
policy_df = build_policy_features(
    policy_df,
    start_col="STAYHOME",
    end_col="END_STHM",
    prefix="stayhome"
)

# use function for closed business mandates
policy_df = build_policy_features(
    policy_df,
    start_col="CLBSNS",
    end_col="END_BSNS",
    prefix="clbsns"
)


# c
policy_df["FM_ALL_dt"] = pd.to_datetime(
    policy_df["FM_ALL"],
    format="%m/%d/%Y",
    errors="coerce"
)

# earliest mask mandate date
earliest_fm_all = policy_df["FM_ALL_dt"].min()

# days since earliest adoption
policy_df["FM_ALL_days_since_earliest_adoption"] = (
    policy_df["FM_ALL_dt"] - earliest_fm_all
).dt.days

# states with no mandate -> 0
policy_df["FM_ALL_days_since_earliest_adoption"] = (
    policy_df["FM_ALL_days_since_earliest_adoption"].fillna(-1)
)


keep_cols = [
    "STATE",
    "FMNOENF",
    "AGEPRIORITY", 
    "stayhome_days_since_earliest_adoption",
    "stayhome_duration_days",
    "clbsns_days_since_earliest_adoption",
    "clbsns_duration_days",
    "FM_ALL_days_since_earliest_adoption"

]

policy_df = policy_df[keep_cols]

policy_df.to_csv(
    "data/processed/clean_policy.csv",
    index=False
)