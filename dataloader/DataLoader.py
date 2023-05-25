from dataclasses import dataclass
import numpy as np
import pandas as pd
import time


@dataclass
class Spectrum:
    xdata: np.ndarray
    ydata: np.ndarray
    device: str
    abs_path_raw: str
    abs_path_ref: str
    calibration: list
    description: list
    fitting: list[float]
    fitting_function: str
    fitting_range: list[flost]
    fitting_values: list[float]
    color: str = 'black'
    linestyle: str = 'solid'
    y_shift: float = 0
    y_times: float = 1
    highlight: bool = False

    def __post_init__(self):
        if self.xdata.shape[0] == 1015:
            self.device = 'Renishaw'
        elif self.xdata.shape[0] == 1024:
            self.device = 'Andor'
        elif self.xdata.shape[0] == 3648:
            self.device = 'CCS'
        else:
            self.device = 'Unknown'

    def reset_appearance(self) -> None:
        self.color = 'black'
        self.linestyle = 'solid'
        self.y_shift = 0
        self.y_times = 1
        self.highlight = False


def extract_keyword(lines: list[str], keyword: str) -> str | None:
    def process(s: str) -> list[str]:
        s = s.strip('# ')
        s = s.strip('\n')
        return s.split(': ')

    for line in lines:
        if keyword in line:
            name, *value = process(line)
            value = ': '.join(value)
            break
    else:
        value = None

    if value == '':
        value = None

    return value


def find_skip(lines: list[str]) -> int:
    numeric_str_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', ',', '-', '\t', '\n']
    for i, line in enumerate(lines):
        for s in line:
            if s not in numeric_str_list:  # if there is a non-numeric character
                break
        else:  # if all the character is numeric
            break
    else:  # if all the lines are non-numeric
        i = -1
    return i


def find_sep(filename: str, skip_rows: int) -> str:
    sep_list = ['\t', ',', '   ']
    num_cols_list = []
    for sep in sep_list:
        df = pd.read_csv(filename, sep=sep, skiprows=skip_rows, header=None)
        num_cols_list.append(df.shape[1])
    return sep_list[num_cols_list.index(max(num_cols_list))]


class DataLoader:
    def __init__(self, filename: str = None, filenames: list[str] = None):
        self.spec_dict: dict[str: Spectrum] = {}

        if filename is not None:
            self.load_file(filename)
        if filenames is not None:
            self.load_files(filenames)

    def load_file(self, filename: str) -> None:
        if filename in self.spec_dict.keys():
            print(f'このファイルは既に読み込まれています．：{filename}')
            return

        with open(filename, 'r') as f:
            lines = f.readlines()

        skip_rows = find_skip(lines)
        if skip_rows == -1:
            raise ValueError('No numeric data found. Check the input file again.')

        spectrum_dict = {}
        for keyword in ['abs_path_raw', 'abs_path_ref', 'calibration', 'description', 'fitting_function', 'fitting_range', 'fitting_values', 'device']:
            value = extract_keyword(lines, keyword)

            if keyword == 'description':
                if value is None:  # if it is the raw data
                    value = lines[:skip_rows]
            elif keyword == 'fitting_range':
                if value is not None:
                    value = list(map(float, value.split()))
                else:
                    value = []
            elif keyword == 'fitting_values':
                if value is not None:
                    value = list(map(float, value.split()))
                else:
                    value = []
            elif keyword == 'abs_path_raw':
                if value is None:
                    value = filename

            spectrum_dict[keyword] = value

        sep = find_sep(filename, skip_rows)
        df = pd.read_csv(filename, sep=sep, skiprows=skip_rows, header=None)
        if df.shape[1] == 1:
            spectrum_dict['xdata'] = np.arange(1, df.shape[0] + 1)
            spectrum_dict['ydata'] = df.iloc[:, 0].values
        else:
            spectrum_dict['xdata'] = df.iloc[:, 0].values
            spectrum_dict['ydata'] = df.iloc[:, 1].values

        self.spec_dict[filename] = Spectrum(**spectrum_dict)

    def load_files(self, filenames: list[str]) -> None:
        for filename in filenames:
            self.load_file(filename)

    def concat_spec(self) -> pd.DataFrame:
        xdata_list = []
        ydata_list = []
        for spec in self.spec_dict.values():
            xdata_list.append(spec.xdata)
            ydata_list.append(spec.ydata)
        xdata_all = np.hstack(xdata_list)
        ydata_all = np.hstack(ydata_list)
        data = np.vstack([xdata_all, ydata_all]).T
        df = pd.DataFrame(data=data, columns=['x', 'y'])
        return df

    def reset_option(self) -> None:
        for spec in self.spec_dict.values():
            spec.reset_appearance()

    def delete_file(self, filename: str) -> None:
        del self.spec_dict[filename]

    def delete_files(self, filenames: list[str]) -> None:
        for filename in filenames:
            self.delete_file(filename)

    def reset_highlight(self) -> None:
        for spec in self.spec_dict.values():
            spec.highlight = False

    def save(self, filename: str, filename_as: str = None) -> None:
        spec = self.spec_dict[filename]
        data = np.vstack((spec.xdata.T, spec.ydata.T)).T
        if filename_as is None:
            filename_as = f'{".".join(filename.split(".")[:-1])}_{time.time()}.{filename.split(".")[-1]}'
        with open(filename_as, 'w') as f:
            f.write(f'# abs_path_raw: {spec.abs_path_raw}\n')
            f.write(f'# abs_path_ref: {spec.abs_path_ref}\n')
            f.write(f'# calibration: {spec.calibration}\n')
            f.write(f'# device: {spec.device}\n')
            f.write(f'# description: {spec.description}\n')
            f.write(f'# fitting_function: {spec.fitting_function}\n')
            f.write(f'# fitting_range: {spec.fitting_range.__str__().replace("[", "").replace("]", "")}\n')
            f.write(f'# fitting_values: {spec.fitting_values.__str__().replace("[", "").replace("]", "")}\n\n')

            for x, y in data:
                f.write(f'{x},{y}\n')
