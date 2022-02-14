import pandas as pd

normal_data = pd.read_csv("../Data/HCRL_Car-Hacking_Dataset/normal_data_set/normal_time_between.csv") 
spoof_data = pd.read_csv("../Data/HCRL_Car-Hacking_Dataset/spoofing_data_set/spoofing_time_between.csv") 

missing_id = normal_data.loc[normal_data['ID'] == "02b0"]
missing_id["Class"] = 0.0
print(len(missing_id))

spoof_data = pd.concat([spoof_data, missing_id])
spoof_data = spoof_data.sample(frac=1).reset_index(drop=True)   #shuffle the order

spoof_data.to_csv("spoofing_time_between.csv")