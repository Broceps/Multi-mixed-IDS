import pandas as pd

"""
Clean data for the normal dataset by:
    - Fixing IDs to map against a numeric value, since theyre not so many unique IDs
    - Calculating the average of each ID's interval, i.e average time between repeated ID
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

# #-----for testing - smaller set goes faster-----
# df = df[10000:500000]
# df = df.reset_index(drop=True)
# #----------


id_map = {}  # the IDs that will be used for the ML training

for i in unique_ids:    #ugly solution but temporary because all code is built around the dictionary
    id_map[i] = i

print(id_map)

for i, row in df.iterrows():
    df.at[i, "ID"] = id_map.get(df.at[i, "ID"])
    # df.at[i,"Timestamp"] = df.at[i,"Timestamp"].partition('.')[0] #round timestamps to seconds only

# test
df.to_csv("check_for_null.csv")


reference_map = {}  # latest timestamp of each ID
# will be incremented for all time-intervals then divided to calculate average
occourance_map = {}
# to calculate occourance of IDs to divide with the result in occourance map
sum_of_ids_map = {}


data = []

for i in id_map:  # init the dictionary to counter 0 of all ids
    occourance_map[id_map[i]] = 0
    sum_of_ids_map[id_map[i]] = 0

for i, row in df.iterrows():  # go throug all rows and calculate intervals
    temp_id = df.at[i, "ID"]
    if temp_id in id_map.values():
        temp_timestamp = df.at[i, "Timestamp"]
        if temp_id in reference_map:  # calculate the difference from last timestamp into occourance_map and update latest timestamp
            diff = (float(temp_timestamp) - float(reference_map.get(temp_id)))
            occourance_map[temp_id] += diff
            sum_of_ids_map[temp_id] += 1
            reference_map[temp_id] = temp_timestamp
        # add the timestamp to reference_map (will only happen for the first occourance of all IDs)
        else:
            reference_map[temp_id] = temp_timestamp
            sum_of_ids_map[temp_id] += 1

print(occourance_map)
print(sum_of_ids_map)

# calculate the average
for key in id_map:
    map_id = id_map.get(key)
    if sum_of_ids_map[map_id] == 0:
        print("\n")#ignore none-occouring messages...
    else:
        avg = (occourance_map[map_id]/sum_of_ids_map[map_id])
        data.append([map_id, avg])

# new df to store results in
new_df = pd.DataFrame(data=data, columns=["ID", "Interval"])

new_df.to_csv("normal_average_time_between.csv", index=False)
