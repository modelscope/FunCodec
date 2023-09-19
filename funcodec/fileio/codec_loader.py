import kaldiio
import numpy as np
import json


class CodecLoader:
    def __init__(
            self,
            filepath,
            quant_groups=32,
            file_type="ark"
    ):
        self.filepath = filepath
        self.quant_groups = quant_groups
        self.file_type = file_type

        self.data_list = []
        self.data_dict = {}
        for line in open(filepath, "rt"):
            uttid, data = line.strip().split(maxsplit=1)
            self.data_dict[uttid] = data
            self.data_list.append(uttid)

    def __iter__(self):
        for uttid in self.data_list:
            yield self[uttid]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, uttid) -> np.ndarray:
        if self.file_type == "ark":
            data = kaldiio.load_mat(self.data_dict[uttid])
            data = np.reshape(data, [self.quant_groups, -1]).T
        elif self.file_type == "jsonl":
            data = json.loads(self.data_dict[uttid])
            data = np.array(data)
        else:
            raise NotImplementedError(f"file type must be in [ark, jsonl] rather than {self.file_type}")
        return data
