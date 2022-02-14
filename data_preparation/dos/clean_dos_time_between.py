import pandas as pd

"""
Clean data for the DoS dataset by:
    - Calculating the interval of each ID
    - Converting Flag to Class, 0 (R) for normal and 1 (T) for injected message
Then saving it into a new csv-file
"""


# Cleaning the normal_state and transforming id's to numbers since theyre not so many unique ones.
fields = ["Timestamp","ID","DLC","0","1","2","3","4","5","6","7","Flag"]


df = pd.read_csv("../Data/HCRL_Car-Hacking_Dataset/raw_data/DoS_dataset.csv", names=fields).astype(str)
df = df[df.Timestamp != "nan"]
df = df[df.ID != "nan"]
# sort so they always get in same order for mapping ids across datasets
unique_ids = set(df.ID)
unique_ids = sorted(unique_ids)

# #-----for testing - smaller set goes faster-----
# df = df[49500:50000]
# df = df.reset_index(drop=True)
# #----------


id_map = {}  # the IDs that will be used for the ML training

for i in unique_ids:    #ugly solution but temporary because all code is built around the dictionary
    id_map[i] = i

for i, row in df.iterrows():
    df.at[i, "ID"] = id_map.get(df.at[i, "ID"])

reference_map = {}  # latest timestamp of each ID
data = []

for i, row in df.iterrows():  # go throug all rows and calculate intervals
    temp_id = df.at[i, "ID"]
    flag = df.at[i, "Flag"]
    if flag == "R":
        flag = 0
    elif flag == "T":
        flag = 1
    if temp_id in id_map.values():
        temp_timestamp = df.at[i, "Timestamp"]
        if temp_id in reference_map:  # calculate the difference from last timestamp into occourance_map and update latest timestamp
            diff = (float(temp_timestamp) - float(reference_map.get(temp_id)))
            data.append([temp_id, diff, flag])
            reference_map[temp_id] = temp_timestamp
        else:  # add the timestamp to reference_map (will only happen for the first occourance of all IDs)
            reference_map[temp_id] = temp_timestamp


# new df to store results in
new_df = pd.DataFrame(data=data, columns=["ID", "Interval","Class"])

new_df.to_csv("dos_time_between.csv", index=False)
