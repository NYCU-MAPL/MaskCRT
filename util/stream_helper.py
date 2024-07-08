import numpy as np

_MAGIC_VALUE_SEP = b'\x46\xE2\x84\x92'

class BitStreamIO:
    """BitStreamIO for Video/Image Compression"""

    def __init__(self, file, mode):
        self.file = file
        self.mode = mode
        self.status = 'open'

        self.strings = b''
        self.streams = list()
        self.shape_strings = list()

    def __len__(self):
        assert self.status == 'open', self.status
        return 1 + np.sum(list(map(len, self.streams+self.shape_strings))) + 4 * (len(self.streams)+len(self.shape_strings))

    @staticmethod
    def shape2string(shape):
        assert len(shape) == 4 and shape[0] == 1, shape
        assert shape[1] < 2 ** 16, shape
        assert shape[2] < 2 ** 16, shape
        assert shape[3] < 2 ** 16, shape
        return np.uint16(shape[1]).tobytes() + np.uint16(shape[2]).tobytes() + np.uint16(shape[3]).tobytes()

    @staticmethod
    def string2shape(string):
        return (1, np.frombuffer(string[0:2], np.uint16)[0],
                np.frombuffer(string[2:4], np.uint16)[0],
                np.frombuffer(string[4:6], np.uint16)[0])

    def write(self, stream_list, shape_list):
        assert self.mode == 'w', self.mode
        self.streams += stream_list
        for shape in shape_list:
            self.shape_strings.append(self.shape2string(shape))

    def read_file(self):
        assert self.mode == 'r', self.mode
        strings = b''
        with open(self.file, 'rb') as f:
            line = f.readline()
            while line:
                strings += line
                line = f.readline()

        self.strings = strings.split(_MAGIC_VALUE_SEP)

        shape_num = int(self.strings[0][0]) // 16
        self.streams, self.shapes = self.strings[shape_num+1:], []
        for shape_strings in self.strings[1:shape_num+1]:
            self.shapes.append(self.string2shape(shape_strings))

        return self.streams, self.shapes

    def read(self, n=1):
        if len(self.strings) == 0:
            self.read_file()

        streams, shapes = [], []
        if len(self.shapes) < n:
            return [], []

        for _ in range(n):
            streams.append(self.streams.pop(0))
            shapes.append(self.shapes.pop(0))

        return streams, shapes

    def split(self, split_size_or_sections):
        if len(self.strings) == 0:
            self.read_file()
        assert len(self.streams) == len(self.shapes)

        if isinstance(split_size_or_sections, int):
            n = split_size_or_sections
            _len = len(self.shapes)
            assert n <= len(self.shapes), (n, len(self.shapes))
            split_size_or_sections = [min(i, n)
                                      for i in range(_len, -1, -n) if i]

        # print(len(self.shapes), split_size_or_sections)
        for n in split_size_or_sections:
            assert n <= len(self.shapes), (n, len(self.shapes))
            ret = self.read(n)
            if len(ret[0]) == 0:
                break
            yield ret

    def chunk(self, chunks):
        if len(self.strings) == 0:
            self.read_file()

        _len = len(self.shapes)
        n = int(np.ceil(_len/chunks))

        return self.split(n)

    def flush(self):
        raise NotImplementedError()

    def close(self):
        assert self.status == 'open', self.status
        if self.mode == 'w':
            shape_num = len(self.shape_strings)
            stream_num = len(self.streams)

            strings = [np.uint8((shape_num << 4) + stream_num).tobytes()]
            strings += self.shape_strings + self.streams

            with open(self.file, 'wb') as f:
                for string in strings[:-1]:
                    f.write(string+_MAGIC_VALUE_SEP)
                f.write(strings[-1])
            del self.streams, self.shape_strings
        else:
            del self.strings, self.streams, self.shapes

        self.status = 'close'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()