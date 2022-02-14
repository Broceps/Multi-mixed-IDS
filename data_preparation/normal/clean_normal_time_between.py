import pandas as pd

"""
Clean data for the normal dataset by:
    - Calculating the interval of each ID
    - Remove "null"-IDs, for normal dataset, that is last ID with key "nan"
Then saving it into a new csv-file
"""


# Cleaning the normal_state and transforming id's to numbers since theyre not so many unique ones.
fields = ["Timestamp", "ID"]
df = pd.read_csv("../Data/HCRL_Car-Hacking_Dataset/raw_data/new_normal_state.csv",
                 skipinitialspace=True, usecols=fields).astype(str)
df = df[df.Timestamp != "nan"]
df = df[df.ID != "nan"]
# sort so they always get in same order for mapping ids across datasets
unique_ids = set(df.ID)
unique_ids = sorted(unique_ids)

#-----for testing - smaller set goes faster-----
df = df[45000:50000]
df = df.reset_index(drop=True)
#----------


id_map = {}  # the IDs that will be used for the ML training

for i in unique_ids:    #ugly solution but temporary because all code is built around the dictionary
    id_map[i] = i[0:4]  #cut off the last zeros


for i, row in df.iterrows():
    df.at[i, "ID"] = id_map.get(df.at[i, "ID"])

reference_map = {}  # latest timestamp of each ID
data = []

for i, row in df.iterrows():  # go throug all rows and calculate intervals
    temp_id = df.at[i, "ID"]
    if temp_id in id_map.values():
        temp_timestamp = df.at[i, "Timestamp"]
        if temp_id in reference_map:  # calculate the difference from last timestamp into occourance_map and update latest timestamp
            diff = (float(temp_timestamp) - float(reference_map.get(temp_id)))
            data.append([temp_id, diff])
            reference_map[temp_id] = temp_timestamp
        else:  # add the timestamp to reference_map (will only happen for the first occourance of all IDs)
            reference_map[temp_id] = temp_timestamp



# new df to store results in
new_df = pd.DataFrame(data=data, columns=["ID", "Interval"])

new_df.to_csv("normal_time_between.csv", index=False)
