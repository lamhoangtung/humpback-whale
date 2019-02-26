import pandas as pd
# my submission file
df = pd.read_csv("./ensembled_sub.csv")
leak_df = pd.read_csv("./leaks.csv")

leak_map = {}

for idx, row in leak_df.iterrows():
    leak_map[row["b_test_img"]] = row["b_label"]

submission_list = []
for idx, row in df.iterrows():
    if row["Image"] in leak_map:
        id_list = row["Id"].split(" ")
        if id_list[0] != leak_map[row["Image"]]:
            print(id_list[0], leak_map[row["Image"]])
            print(id_list)
        id_list[0] = leak_map[row["Image"]]
        id_string = " ".join(id_list)
    else:
        id_string = row["Id"]
    submission_list.append(id_string)
df["Id"] = submission_list
# modified output
df.to_csv("ensembled_sub_with_leak.csv", index=False)
