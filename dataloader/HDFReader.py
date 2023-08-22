import os.path

import h5py


class HDFReader:
    def __init__(self, path):
        if os.path.splitext(path)[1] != '.hdf5':
            raise ValueError('File extension must be .hdf5')
        self.path = path
        self.file = h5py.File(path, 'r')

    def __getitem__(self, item):
        return self.file[item]

    def close(self):
        self.file.close()


class HDFWriter:
    def __init__(self, path):
        if os.path.splitext(path)[1] != '.hdf5':
            raise ValueError('File extension must be .hdf5')
        self.path = path
        self.file = h5py.File(path, 'w')

    def __setitem__(self, key, value):
        self.file[key] = value

    def close(self):
        self.file.close()


class RamanHDFReader(HDFReader):
    def __init__(self, path):
        super().__init__(path)
        self.xdata = self.file['xdata'][:]
        self.spactra = self.file['spectra'][:]
        self.x_start = self.file['x_start'][()]
        self.y_start = self.file['y_start'][()]
        self.x_pad = self.file['x_pad'][()]
        self.y_pad = self.file['y_pad'][()]
        self.x_span = self.file['x_span'][()]
        self.y_span = self.file['y_span'][()]
        self.map_info = {
            'x_start': self.file['x_start'][()],
            'y_start': self.file['y_start'][()],
            'x_pad': self.file['x_pad'][()],
            'y_pad': self.file['y_pad'][()],
            'x_span': self.file['x_span'][()],
            'y_span': self.file['y_span'][()]
        }


class RamanHDFWriter(HDFWriter):
    def __init__(self, path):
        super().__init__(path)

    def create_attr(self, key, value):
        self.file.attrs[key] = value

    def create_dataset(self, name, data):
        self.file.create_dataset(name, data=data)
