import pandas as pd
from pandas import DataFrame


def main():
    file_path = "C:/Users/Acer/Desktop/dataset_with_agr.csv"
    df = DataFrame(pd.read_csv(file_path, encoding='utf-8', sep=';', engine='python'))
    workshop_list = ["007", "053", "009", "042", "043"]
    ws_to_ws_name_dict = get_workshop_dict(workshop_list)

    for data_row_id, data_row in df.iterrows():
        for ws_name in ws_to_ws_name_dict:
            if str(data_row["osn"]).endswith(ws_name):
                df.loc[data_row_id, "agr"] = ws_to_ws_name_dict[ws_name]
                df.loc[data_row_id, "osn"] = df.loc[data_row_id, "osn"].removesuffix(f"-{ws_name}")
                break
            else:
                df.loc[data_row_id, "agr"] = "-1"

    print(df[["osn", "agr"]])

    df.to_csv(file_path.removesuffix(".csv") + "_with_agr.csv", encoding="utf-8", sep=";", index=False)


def get_workshop_dict(ws_list: list) -> dict:
    ws_dict = {}

    for ws_num in ws_list:
        ws_dict[ws_num] = str(int(ws_num))

    return ws_dict


if __name__ == '__main__':
    main()
