from pathlib import Path
import h5py


class HDFReader:
    def __init__(self, p: Path):
        if p.suffix != '.hdf5':
            raise ValueError('File extension must be .hdf5')
        self.path = p
        self.file = h5py.File(p, 'r')

    def __getitem__(self, item):
        return self.file[item]

    def close(self):
        self.file.close()


class HDFWriter:
    def __init__(self, p: Path):
        if p.suffix != '.hdf5':
            raise ValueError('File extension must be .hdf5')
        self.path = p
        self.file = h5py.File(p, 'w')

    def __setitem__(self, key, value):
        self.file[key] = value

    def close(self):
        self.file.close()


class RamanHDFReader(HDFReader):
    def __init__(self, p: Path):
        super().__init__(p)
        self.xdata = self.file['xdata'][:]
        self.spectra = self.file['spectra'][:]

        self.time = self.file.attrs['time']
        self.integration = self.file.attrs['integration']
        self.accumulation = self.file.attrs['accumulation']
        self.pixel_size = self.file.attrs['pixel_size']
        self.shape = self.file.attrs['shape'][:]
        self.map_info = {
            'x_start': self.file.attrs['x_start'],
            'y_start': self.file.attrs['y_start'],
            'x_pad': self.file.attrs['x_pad'],
            'y_pad': self.file.attrs['y_pad'],
            'x_span': self.file.attrs['x_span'],
            'y_span': self.file.attrs['y_span']
        }


class RamanHDFWriter(HDFWriter):
    def __init__(self, p: Path):
        super().__init__(p)

    def create_attr(self, key, value):
        self.file.attrs[key] = value

    def create_dataset(self, name, data):
        self.file.create_dataset(name, data=data)
