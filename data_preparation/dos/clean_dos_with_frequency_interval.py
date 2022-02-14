import pandas as pd

"""
Clean data for the DoS dataset by:
    - Fixing IDs to map against a numeric value, since theyre not so many unique IDs
    - Create a new feature called frequency, for summing the occourance of each ID of every second to get the Hz
    - Remove "null"-IDs, for DoS dataset, that is first ID with key "0000"
    - Adding missing ID "02b0000" (ID = 12) with a null-value (because the IDs will not be correct across datasets otherwise)
Then saving it into a new csv-file
"""


#Cleaning the normal_state and transforming id's to numbers since theyre not so many unique ones.
fields = ["Timestamp","ID","DLC","0","1","2","3","4","5","6","7","Flag"]
df = pd.read_csv("../Data/HCRL_Car-Hacking_Dataset/raw_data/DoS_dataset.csv", names=fields).astype(str)
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
    df.at[i,"ID"] = id_map.get(df.at[i, "ID"])
    df.at[i,"Timestamp"] = df.at[i,"Timestamp"].partition('.')[0] #round timestamps to seconds only


#Calculate the occourence of every ID for every second
reference = df.at[0, "Timestamp"] #First reference point, always check with reference and then sum up when a new second is given
occourance_map = {} #will be reset every new timeframe and result put into new_df
data = []

for i in id_map:        #init the dictionary to counter 0 of all ids
    occourance_map[id_map[i]] = 0

for i, row in df.iterrows():     #go throug all rows
    if df.at[i, "ID"] in id_map.values():
        if (int(reference)-int(df.at[i,"Timestamp"]))!=0: #collect every 1 second
            #new timeframe
            for key in occourance_map:
                data.append([reference, key, occourance_map[key]]) 
                #reset occourances
                occourance_map[key] = 0
            #update timeframe
            reference = df.at[i,"Timestamp"]
        else:
            occourance_map[df.at[i,"ID"]] +=1

#move over the "leftovers" that got stuck between the timeperiods
for key in occourance_map:
    data.append([reference, key, occourance_map[key]])

new_df = pd.DataFrame(data=data,columns=["Timestamp", "ID","Frequency"]) #new df to store results in

new_df.to_csv("dos_frequency_interval.csv", index=False)