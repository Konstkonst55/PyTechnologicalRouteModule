import argparse


def main():
    args_dict = parse_args()
    print(args_dict.items())


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description='Позволяет прогнозировать различные технологические маршруты деталей')
    parser.add_argument(
        '-is_osn', '--is_osn',
        help='Параметр, отвечающий за то, по какому цеху ведется прогнозирование',
        required=True
    )
    parser.add_argument('-n', '--name', help='Наименование', required=True)
    # todo размеры -> габариты
    parser.add_argument('-x', '--x', help='Размер X', required=True)
    parser.add_argument('-y', '--y', help='Размер Y', required=True)
    parser.add_argument('-z', '--z', help='Размер Z', required=True)
    parser.add_argument('-c', '--cg', help='Конструктивная группа (2 цифры)', required=True)
    parser.add_argument('-m', '--mark', help='Марка (первое слово)', required=True)
    parser.add_argument('-s', '--spf', help='Полуфабрикат (первое слово)', required=True)
    parser.add_argument('-t', '--tt', help='Технические требования (через "|")', required=True)
    return vars(parser.parse_args())


if __name__ == '__main__':
    main()
    # todo delete this >>>
    input("---")
