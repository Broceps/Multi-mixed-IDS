import pandas as pd

"""
Clean data for the DOS dataset by:
    - Fixing IDs to map against a numeric value, since theyre not so many unique IDs
    - Create a new feature called frequency, for summing the occourance of each ID
Then saving it into a new csv-file
"""
fields = ["Timestamp","ID","DLC","0","1","2","3","4","5","6","7","Flag"]


df = pd.read_csv("../Data/HCRL_Car-Hacking_Dataset/raw_data/DoS_dataset.csv", names=fields).astype(str)

# #-----for testing - smaller set goes faster-----
# df = df[50000:55000]
# df = df.reset_index(drop=True)
# #----------

unique_ids = set(df.ID)
unique_ids = sorted(unique_ids) #sort so they always get in same order for mapping ids across datasets

id_map = {}         #the IDs that will be used for the ML training 
counter = 1
for i in unique_ids:
    id_map[i] = counter
    counter+=1

for i, row in df.iterrows():
    df.at[i,"ID"] = id_map.get(df.at[i, "ID"])

normal_frequency_map = {}
injection_frequency_map = {}
for i in id_map:        #init the dictionary to counter 0 of all ids
    normal_frequency_map[id_map[i]] = 0
    injection_frequency_map[id_map[i]] = 0


print("\nBEFORE\nnormal: ",normal_frequency_map,"\n\ninjection: ", injection_frequency_map)

for i, row in df.iterrows():     #iterate and calculate frequency and split dataset by message flag (T = injection, R = normal)
    flag = df.at[i, "Flag"]
    if flag == 'R':
        normal_frequency_map[df.at[i, "ID"]]+=1
    elif flag == 'T':
        injection_frequency_map[df.at[i, "ID"]]+=1

print("\nAFTER\n")
print("\nnormal: ",normal_frequency_map)
print("\ninjection: ",injection_frequency_map)

normal_data = []
injection_data = []
for key in normal_frequency_map:
    normal_data.append([key, normal_frequency_map[key]])

for key in injection_frequency_map:
    injection_data.append([key, injection_frequency_map[key]])

columns = ["ID", "frequency"]
df1 = pd.DataFrame(data=normal_data, columns=columns)
df2 = pd.DataFrame(data=injection_data, columns=columns)

df1.to_csv("normal_dos.csv", index=False)
df2.to_csv("injection_dos.csv", index=False)