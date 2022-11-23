import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas


class ExcelDataset(Dataset):
    def __init__(self, filepath=r"file/test.xlsx", sheet_name=0):
        print(f"reading {filepath}, sheet={sheet_name}")

        df = pandas.read_excel(
            # header=0 指定第0行为表头
            # index_col=0 指定第0列为索引
            filepath, header=0, index_col=0,
            names=['id', 'feat1', 'feat2', 'label'],
            dtype={"id": np.int32, "feat1": np.float32, "feat2": np.float32, "label": np.int32}
        )

        print(f"the shape of dataframe is {df.shape}")

        feat = df.iloc[:, 0:2].values
        label = df.iloc[:, 2].values
        self.x = torch.from_numpy(feat)
        self.y = torch.from_numpy(label)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class CsvDataset(Dataset):
    def __init__(self, filepath=r'file/test.csv'):
        df = pandas.read_csv(
            filepath, header=0, index_col=0,
            encoding='utf-8',
            names=['id', 'feat1', 'feat2', 'label'],
            dtype={"feat1": np.float32, "feat2": np.float32, "label": np.int32}
        )

        print(f"the shape of dataframe is {df.shape}")

        feat = df.iloc[:, :3].values
        label = df.iloc[:, 2].values
        self.x = torch.from_numpy(feat)
        self.y = torch.from_numpy(label)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


if __name__ == "__main__":
    excel_dataset = ExcelDataset(sheet_name=0)
    excel_dataloader = DataLoader(excel_dataset, batch_size=8, shuffle=True)
    for idx, (batch_x, batch_y) in enumerate(excel_dataloader):
        print(f"idx: {idx}, {batch_x.shape}, {batch_y.shape}")
        print(batch_x)
        print(batch_y)
    csv_dataset = CsvDataset()
    csv_dataloader = DataLoader(excel_dataset, batch_size=8, shuffle=True)
    for idx, (batch_x, batch_y) in enumerate(excel_dataloader):
        print(f"idx: {idx}, {batch_x.shape}, {batch_y.shape}")
        print(batch_y)
