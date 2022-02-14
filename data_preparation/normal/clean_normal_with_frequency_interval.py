import pandas as pd

"""
Clean data for the normal dataset by:
    - Removing datapoints with no ID or no Timestamp.
    - Create a new feature called frequency, for summing the occourance of each ID of every second and then
        dividing by 1000 to get it in ms.
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
    id_map[i] = i[0:4]  #cut off the last zeros

print(id_map)

for i, row in df.iterrows():
    df.at[i, "ID"] = id_map.get(df.at[i, "ID"])
    df.at[i, "Timestamp"] = df.at[i, "Timestamp"].partition(
        '.')[0]  # round timestamps to seconds only

# test
df.to_csv("check_for_null.csv")

# Calculate the occourence of every ID for every second
# First reference point, always check with reference and then sum up when a new second is given
reference = df.at[0, "Timestamp"]
occourance_map = {}  # will be reset every new timeframe and result put into new_df
data = []

for i in id_map:  # init the dictionary to counter 0 of all ids
    occourance_map[id_map[i]] = 0

for i, row in df.iterrows():  # go throug all rows
    if df.at[i, "ID"] in id_map.values():
        # collect every 1 second
        if (int(reference)-int(df.at[i, "Timestamp"])) != 0:
            # new timeframe
            for key in occourance_map:
                # divide by 1000 to get frequency in ms
                data.append([reference, key, occourance_map[key]])
                # reset occourances
                occourance_map[key] = 0
            # update timeframe
            reference = df.at[i, "Timestamp"]
        else:
            occourance_map[df.at[i, "ID"]] += 1

# move over the "leftovers" that got stuck between the timeperiods
for key in occourance_map:
    # divide by 1000 to get frequency in ms
    data.append([reference, key, occourance_map[key]])

# new df to store results in
new_df = pd.DataFrame(data=data, columns=["Timestamp", "ID", "Frequency"])

new_df.to_csv("normal_frequency_interval.csv", index=False)
