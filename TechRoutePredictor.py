import argparse
import os
import sys
from operator import concat
from pathlib import Path

import pandas as pd
import pkg_resources
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

MODEL_OSN_PATH: str = "data/models/osn/"
MODEL_OSN_FILE_NAME: str = "model_osn_details.joblib"
ENCODER_OSN_PATH: str = "data/encoders/osn/"
ENCODER_OSN_FILE_NAME: str = "label_encoders_osn_dict.joblib"
MODEL_USL_PATH: str = "data/models/usl/"
MODEL_USL_FILE_NAME: str = "model_usl_details.joblib"
ENCODER_USL_PATH: str = "data/encoders/usl/"
ENCODER_USL_FILE_NAME: str = "label_encoders_usl_dict.joblib"

process_type_dict = {
    "predict": ["name", "gab", "osn", "cg", "mark", "spf", "tt", "agr", "pt"],
    "fit": ["fit", "osn"]
}

df_columns = ['name', 'gab', 'cg', 'mark', 'spf', 'tt', 'agr']
osn_col_name = 'osn'
usl_col_name = 'usl'


class DataProcessor:
    def __init__(self):
        self.le_dict_osn = get_le(ENCODER_OSN_PATH, ENCODER_OSN_FILE_NAME)
        self.le_dict_usl = get_le(ENCODER_USL_PATH, ENCODER_USL_FILE_NAME)

    def process_data(self, file_name: str, predict_col_name: str):
        try:
            dataset = read_data_file(file_name)

            if 'uid' in dataset.columns:
                dataset = dataset.drop('uid', axis=1)

            if predict_col_name == "osn":
                dataset = self.fit_transform_osn_data(dataset)
            elif predict_col_name == "usl":
                dataset = self.fit_transform_usl_data(dataset)
            else:
                raise ValueError(f"Ошибка в наименовании прогнозируемой колонки {predict_col_name}")

            trg_data = dataset[[predict_col_name]].values.ravel()
            trn_data = dataset.drop(predict_col_name, axis=1)

            return train_test_split(trn_data, trg_data, test_size=0.1)

        except KeyError as ke:
            raise KeyError(f"Проверьте входной файл {str(ke)}")
        except ValueError as ve:
            raise ValueError(str(ve))

    def fit_transform_osn_data(self, df: DataFrame) -> DataFrame:
        for col in df.columns:
            first = df.loc[df.index[0], col]
            if isinstance(first, str) or first == -1:
                self.le_dict_osn[col] = LabelEncoder()
                df[col] = self.le_dict_osn[col].fit_transform(df.astype(str).__getattr__(col))
        save(self.le_dict_osn, ENCODER_OSN_PATH, ENCODER_OSN_FILE_NAME)
        return df

    def transform_osn_data(self, df: DataFrame) -> DataFrame:
        try:
            for col in df.columns:
                first = df.loc[df.index[0], col]
                if isinstance(first, str) or first == -1:
                    df[col] = self.le_dict_osn[col].transform(df.astype(str).__getattr__(col))
            return df

        except ValueError as ve:
            raise ValueError(f"Неизвестные входные данные {str(ve)}")
        except KeyError as ke:
            raise KeyError(f"Неизвестные входные данные {str(ke)}")

    def fit_transform_usl_data(self, df: DataFrame) -> DataFrame:
        for col in df.columns:
            first = df.loc[df.index[0], col]
            if isinstance(first, str) or first == -1:
                self.le_dict_usl[col] = LabelEncoder()
                df[col] = self.le_dict_usl[col].fit_transform(df.astype(str).__getattr__(col))
        save(self.le_dict_usl, ENCODER_USL_PATH, ENCODER_USL_FILE_NAME)
        return df

    def transform_usl_data(self, df: DataFrame) -> DataFrame:
        try:
            for col in df.columns:
                first = df.loc[df.index[0], col]
                if isinstance(first, str) or first == -1:
                    df[col] = self.le_dict_usl[col].transform(df.astype(str).__getattr__(col))
            return df

        except ValueError as ve:
            raise ValueError(f"Неизвестные входные данные {str(ve)}")
        except KeyError as ke:
            raise KeyError(f"Неизвестные входные данные {str(ke)}")


class ModelProcessor:
    def __init__(self):
        self.rfr_model_osn = get_rfr_model(MODEL_OSN_PATH, MODEL_OSN_FILE_NAME)
        self.rfr_model_usl = get_rfr_model(MODEL_USL_PATH, MODEL_USL_FILE_NAME)

    def fit_osn_model(self, x_trn, y_trn, x_test, y_test):  # todo тестовые данные использовать в проверке качества
        self.rfr_model_osn.fit(x_trn, y_trn)
        save(self.rfr_model_osn, MODEL_OSN_PATH, MODEL_OSN_FILE_NAME)

    def fit_usl_model(self, x_trn, y_trn, x_test, y_test):  # todo тестовые данные использовать в проверке качества
        self.rfr_model_usl.fit(x_trn, y_trn)
        save(self.rfr_model_usl, MODEL_USL_PATH, MODEL_USL_FILE_NAME)

    def predict_data(self, data: DataFrame, predict_col_name: str, data_processor: DataProcessor) -> DataFrame:
        if predict_col_name == "osn":
            predict = self.rfr_model_osn.predict(data_processor.transform_osn_data(data))
        elif predict_col_name == "usl":
            predict = self.rfr_model_usl.predict(data_processor.transform_usl_data(data))
        else:
            raise ValueError(f"Ошибка в наименовании прогнозируемой колонки {predict_col_name}")

        data.insert(5, predict_col_name, predict)

        return data


def main():
    args_dict = parse_args()
    print(args_dict.items())
    data_processor = DataProcessor()
    model_processor = ModelProcessor()

    if check_dict(args_dict, process_type_dict["fit"]):
        fit_model(args_dict, model_processor, data_processor)
        return

    if check_dict(args_dict, process_type_dict["predict"]):
        predict_data(args_dict, model_processor, data_processor)
        return


def check_dict(dic: dict, args: list[str]) -> bool:
    for arg in args:
        if not dic.__contains__(arg):
            return False

        if dic[arg] is None:
            return False
    return True


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description='Позволяет прогнозировать различные технологические маршруты деталей')

    parser.add_argument('-f', '--fit', help='Путь к файлу .csv для обучения модели')
    parser.add_argument(
        '-osn', '--osn',  # --osn or --no-osn
        help='Параметр, отвечающий за то, по какому цеху ведется прогнозирование (по умолчанию True)',
        required=False,
        action=argparse.BooleanOptionalAction,
        default=True
    )
    parser.add_argument(
        '-pt', '--pt',
        help='Путь для сохранения текстового файла с выводом (с наименованием файла)',
        required=False
    )
    parser.add_argument('-n', '--name', help='Наименование (первое слово)', required=False)
    parser.add_argument('-g', '--gab', help='Габариты (0-4)', type=float, required=False)
    parser.add_argument('-c', '--cg', help='Конструктивная группа (2 цифры)', type=float, required=False)
    parser.add_argument('-m', '--mark', help='Марка (первое слово)', required=False)
    parser.add_argument('-s', '--spf', help='Полуфабрикат (первое слово)', required=False)
    parser.add_argument('-t', '--tt', help='Технические требования - передается как путь до файла с тт', required=False)
    parser.add_argument('-a', '--agr', help='Тип агрегата (0-4)', required=False)

    return vars(parser.parse_args())


def fit_model(args_dict: dict, mp: ModelProcessor, dp: DataProcessor):
    try:
        file_name = args_dict["fit"]

        if args_dict["osn"]:
            x_trn, x_test, y_trn, y_test = dp.process_data(file_name, osn_col_name)
            mp.fit_osn_model(x_trn, y_trn, x_test, y_test)
        else:
            x_trn, x_test, y_trn, y_test = dp.process_data(file_name, usl_col_name)
            mp.fit_usl_model(x_trn, y_trn, x_test, y_test)
    except KeyError as ke:
        print(str(ke))
        input("Press Enter...")
    except FileExistsError as fee:
        print(str(fee))
        input("Press Enter...")
    except ValueError as ve:
        print(str(ve))
        input("Press Enter...")


def get_data_from_args(args_dict: dict):
    return DataFrame(
        [[args_dict["name"],
          args_dict["gab"],
          args_dict["cg"],
          args_dict["mark"],
          args_dict["spf"],
          read_txt(args_dict["tt"]),
          args_dict["agr"]]],
        columns=df_columns
    )


def predict_data(args_dict: dict, mp: ModelProcessor, dp: DataProcessor):
    try:
        if args_dict["osn"]:
            if not hasattr(mp.rfr_model_osn, "estimators_"):
                raise AttributeError("Сначала необходимо обучить модель osn")

            predict_df = mp.predict_data(get_data_from_args(args_dict), osn_col_name, dp)

            write_txt(
                args_dict["pt"],
                str(
                    dp.le_dict_osn[osn_col_name]
                    .inverse_transform(
                        predict_df.astype(int).__getattr__(osn_col_name)
                    )[0]  # Выводится только первая запись (записи подаются одной строкой)
                )
            )
        else:
            if not hasattr(mp.rfr_model_usl, "estimators_"):
                raise AttributeError("Сначала необходимо обучить модель usl")

            predict_df = mp.predict_data(get_data_from_args(args_dict), usl_col_name, dp)

            write_txt(
                args_dict["pt"],
                str(
                    dp.le_dict_usl[usl_col_name]
                    .inverse_transform(
                        predict_df.astype(int).__getattr__(usl_col_name)
                    )[0]  # Выводится только первая запись (записи подаются одной строкой)
                )
            )

    except AttributeError as ae:
        print(str(ae))
        input("Press Enter...")
    except KeyError as ke:
        print(str(ke))
        input("Press Enter...")
    except ValueError as ve:
        print(str(ve))
        input("Press Enter...")


def write_txt(path: str, text: str):
    head_path, file_name = os.path.split(path)

    if Path(head_path).exists() and file_name.endswith(".txt"):
        with(open(path, "w+")) as f:
            f.write(text)
            f.close()


def read_txt(path: str) -> str:
    head_path, file_name = os.path.split(path)

    if Path(head_path).exists() and file_name.endswith(".txt"):
        with(open(path, "r")) as f:
            return f.read()


def show_exception_and_exit(exc_type, exc_value, tb):
    import traceback
    traceback.print_exception(exc_type, exc_value, tb)
    sys.exit(-1)


def get_rfr_model(path: str, name: str) -> RandomForestRegressor:
    full_path = f"{path}{name}"

    if Path(full_path).exists():
        return joblib.load(full_path)
    else:
        print("get_rfr_model вернул RandomForestRegressor")
        return RandomForestRegressor()


def get_le(path: str, name: str) -> dict:
    full_path = f"{path}{name}"

    if Path(full_path).exists():
        return joblib.load(full_path)
    else:
        print("get_le вернул {}")
        return {}


def save(value, path: str, name: str):
    if Path(path).exists():
        joblib.dump(value, concat(path, name))
    else:
        raise FileExistsError(f"Путь {Path(path).absolute()}, указанный при сохранении файла {name} отсутствует")


def read_data_file(file_name: str) -> DataFrame:
    return DataFrame(pd.read_csv(file_name, encoding='utf-8', sep=';', engine='python'))


if __name__ == '__main__':
    MODEL_OSN_PATH = pkg_resources.resource_filename(__name__, "data/models/osn/")
    ENCODER_OSN_PATH = pkg_resources.resource_filename(__name__, "data/encoders/osn/")
    MODEL_USL_PATH = pkg_resources.resource_filename(__name__, "data/models/usl/")
    ENCODER_USL_PATH = pkg_resources.resource_filename(__name__, "data/encoders/usl/")
    main()
    # todo delete this >>>
    input("Press Enter...")
