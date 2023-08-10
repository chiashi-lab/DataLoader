import re
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
    fitting_function: str
    fitting_range: list[float]
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
    pattern = re.compile(f'^# {keyword}: (.*)$')
    matched = [pattern.match(line).group(1) for line in lines if pattern.match(line)]

    if len(matched) == 0:
        value = None
    elif len(matched) == 1:
        value = matched[0]
    else:
        raise ValueError(f'Keyword {keyword} is duplicated.')

    if value == '':
        value = None

    return value


NUMERIC_PATTERN = r'[+-]?\d+(?:\.\d+)?'
SEP_PATTERN = r'[ ,\t]+'
IS_NUMERIC = re.compile(NUMERIC_PATTERN)
IS_NUMERIC_SEP_NUMERIC = re.compile(fr'^({NUMERIC_PATTERN})({SEP_PATTERN})({NUMERIC_PATTERN})') # for finding the separator
IS_NUMERIC_ROW = re.compile(f'^({NUMERIC_PATTERN}{SEP_PATTERN})*{NUMERIC_PATTERN}$')


def find_skip(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if IS_NUMERIC_ROW.match(line):
            break
    else:  # if all lines are non-numeric
        i = -1
    return i


def find_sep(line):
    matched = IS_NUMERIC_SEP_NUMERIC.match(line)
    if matched is None:
        return ''
    else:
        return matched.group(2)


class DataLoader:
    def __init__(self, filename: str = None, filenames: list[str] = None):
        self.spec_dict: dict[str: Spectrum] = {}

        if filename is not None:
            self.load_file(filename)
        if filenames is not None:
            self.load_files(filenames)

    def load_file(self, filename: str) -> bool:
        if filename in self.spec_dict.keys():
            print(f'このファイルは既に読み込まれています：{filename}')
            return False
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            print('非対応のファイル形式です')
            return False

        skiprows = find_skip(lines)
        if skiprows == -1:
            raise ValueError('No numeric data found. Check the input file again.')
        skipfooter = find_skip(lines[::-1])

        spectrum_dict = {}
        for keyword in ['abs_path_raw', 'abs_path_ref', 'calibration', 'description', 'fitting_function', 'fitting_range', 'fitting_values', 'device']:
            value = extract_keyword(lines, keyword)

            if keyword == 'abs_path_raw':
                if value is None:
                    value = filename
            elif keyword == 'description':
                if spectrum_dict.get('abs_path_raw') == filename:  # if itself is the raw data
                    value = lines[:skiprows]
            elif keyword == 'fitting_range':
                if value is not None:
                    value = list(map(float, value.split(', ')))
                else:
                    value = []
            elif keyword == 'fitting_values':
                if value is not None:
                    value = list(map(float, value.split(', ')))
                else:
                    value = []

            spectrum_dict[keyword] = value

        sep = find_sep(lines[skiprows])
        df = pd.read_csv(filename, engine='python', encoding='cp932', sep=sep, skiprows=skiprows, skipfooter=skipfooter, header=None)
        if df.shape[1] == 1:
            spectrum_dict['xdata'] = np.arange(1, df.shape[0] + 1)
            spectrum_dict['ydata'] = df.iloc[:, 0].values
            self.spec_dict[filename] = Spectrum(**spectrum_dict)
            print('Only one column is found. The first column is used as ydata.')
        elif df.shape[1] == 2:
            spectrum_dict['xdata'] = df.iloc[:, 0].values
            spectrum_dict['ydata'] = df.iloc[:, 1:].values
            self.spec_dict[filename] = Spectrum(**spectrum_dict)
        else:
            spectrum_dict['xdata'] = df.iloc[:, 0].values
            spectrum_dict['ydata'] = df.iloc[:, 1:].values.T
            for i, ydata in enumerate(spectrum_dict['ydata']):
                tmp_dict = spectrum_dict.copy()
                tmp_dict['ydata'] = ydata
                self.spec_dict[f'{filename}({i})'] = Spectrum(**tmp_dict)

        return True

    def load_files(self, filenames: list[str]) -> dict:
        ok_dict = {}
        for filename in filenames:
            ok = self.load_file(filename)
            ok_dict[filename] = ok
        return ok_dict

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
            f.write(f'# abs_path_ref: {spec.abs_path_ref if spec.abs_path_ref else ""}\n')
            f.write(f'# calibration: {spec.calibration if spec.calibration else ""}\n')
            f.write(f'# device: {spec.device}\n')
            f.write(f'# description: {spec.description if spec.description else ""}\n')
            f.write(f'# fitting_function: {spec.fitting_function if spec.fitting_function else ""}\n')
            f.write(f'# fitting_range: {",".join(spec.fitting_range) if spec.fitting_range else ""}\n')
            f.write(f'# fitting_values: {", ".join(spec.fitting_values) if spec.fitting_values else ""}\n\n')

            for x, y in data:
                f.write(f'{x},{y}\n')


def test():
    # IS_NUMERIC.match
    def is_numeric(s: str) -> bool:
        return IS_NUMERIC_ROW.match(s) is not None
    assert is_numeric('1')
    assert is_numeric('1, 2\n')
    assert is_numeric('1, 2, 3\n')
    assert is_numeric('1, 2, 3, 4, 5, 6')
    assert is_numeric('-111,+100')
    assert not is_numeric('a')
    assert not is_numeric('1 ')
    assert not is_numeric('1, 2, 3,')
    assert not is_numeric('1, 2e')
    assert not is_numeric('-111,--100')
    print('is_numeric: OK')

    # find_skip
    lines = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    assert find_skip(lines) == -1
    lines = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', '1, 2, 3']
    assert find_skip(lines) == 9
    assert find_skip(lines[::-1]) == 0
    lines = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i\n', '1, 2, 3', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's']
    assert find_skip(lines) == 9
    assert find_skip(lines[::-1]) == 10
    print('find_skip: OK')

    # find_sep
    assert find_sep('1') == ''
    assert find_sep('1,1') == ','
    assert find_sep('1,1,1') == ','
    assert find_sep('1, 1') == ', '
    assert find_sep('-1, -1') == ', '
    assert find_sep('1  -1') == '  '
    assert find_sep('1\t-1') == '\t'
    assert find_sep('1\t-1\n') == '\t'
    print('find_sep: OK')

    # load_file
    loader = DataLoader()
    loader.load_file('test.txt')
    assert loader.spec_dict['test.txt'].abs_path_raw == 'raw.txt'
    assert loader.spec_dict['test.txt'].abs_path_ref == 'ref.txt'
    assert loader.spec_dict['test.txt'].calibration == 'sulfur 1 2 3'
    assert loader.spec_dict['test.txt'].device == 'Unknown'
    assert loader.spec_dict['test.txt'].description is None
    assert loader.spec_dict['test.txt'].xdata[0] == 100
    assert loader.spec_dict['test.txt'].xdata[-1] == 200
    assert loader.spec_dict['test.txt'].ydata[0] == 100
    assert loader.spec_dict['test.txt'].ydata[-1] == -100
    print('load_file: OK')


if __name__ == '__main__':
    test()
