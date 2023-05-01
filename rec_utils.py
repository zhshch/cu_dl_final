import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


PAD = 0
MASK = 1


def mask_list(l1, p=0.8):
    l1 = [a if random.random() < p else MASK for a in l1]
    return l1


def mask_last_elements_list(l1, val_context_size: int = 5):
    l1 = l1[:-val_context_size] + mask_list(l1[-val_context_size:], p=0.5)
    return l1

def map_column(df: pd.DataFrame, col_name: str):
    """
    Maps column values to integers
    :param df:
    :param col_name:
    :return:
    """
    values = sorted(list(df[col_name].unique()))
    mapping = {k: i + 2 for i, k in enumerate(values)}
    inverse_mapping = {v: k for k, v in mapping.items()}

    df[col_name + "_mapped"] = df[col_name].map(mapping)

    return df, mapping, inverse_mapping


def get_context(df: pd.DataFrame, stage: str, context_size: int = 120, val_context_size: int = 5):
    """
    Create a training / validation samples
    Validation samples are the last horizon_size rows
    :param df:
    :param split:
    :param context_size:
    :param val_context_size:
    :return:
    """
    if stage == "train":
        end_index = random.randint(10, df.shape[0] - val_context_size)
    elif stage in ["val", "test"]:
        end_index = df.shape[0]
    else:
        raise ValueError

    start_index = max(0, end_index - context_size)
    context = df[start_index:end_index]
    return context


def pad_arr(arr: np.ndarray, expected_size: int = 30):
    """
    Pad top of array when there is not enough history
    :param arr:
    :param expected_size:
    :return:
    """
    arr = np.pad(arr, [(expected_size - arr.shape[0], 0), (0, 0)], mode="edge")
    return arr


def pad_list(list_integers, history_size: int, pad_val: int = PAD, mode="left"):
    """

    :param list_integers:
    :param history_size:
    :param pad_val:
    :param mode:
    :return:
    """

    if len(list_integers) < history_size:
        if mode == "left":
            list_integers = [pad_val] * (history_size - len(list_integers)) + list_integers
        else:
            list_integers = list_integers + [pad_val] * (history_size - len(list_integers))

    return list_integers


def df_to_np(df, expected_size=30):
    arr = np.array(df)
    arr = pad_arr(arr, expected_size=expected_size)
    return arr


def genome_mapping(genome):
    genome.sort_values(by=["movieId", "tagId"], inplace=True)
    movie_genome = genome.groupby("movieId")["relevance"].agg(list).reset_index()
    movie_genome = {a: b for a, b in zip(movie_genome['movieId'], movie_genome['relevance'])}

    return movie_genome


def predict_movie(list_movies, model, movie_to_idx, idx_to_movie):
    ids = [PAD] * (100 - len(list_movies) - 1) + [movie_to_idx[a] for a in list_movies] + [MASK]
    src = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        prediction = model(src)
    masked_pred = prediction[0, -1].numpy()
    sorted_predicted_ids = np.argsort(masked_pred).tolist()[::-1]
    sorted_predicted_ids = [a for a in sorted_predicted_ids if a not in ids]
    return [idx_to_movie[a] for a in sorted_predicted_ids[:30] if a in idx_to_movie]


class BRDataset(Dataset):
    def __init__(self, groups, group_by_df, stage, history_size=100):
        self.group_by_df = group_by_df
        self.groups = groups
        self.stage = stage
        self.history_size = history_size

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        df = self.group_by_df.get_group(group)
        context = get_context(df, stage=self.stage, context_size=self.history_size)
        mapped_items = context["item_id_mapped"].tolist()

        if self.stage == "train":
            mask_items = mask_list(mapped_items)
        else:
            mask_items = mask_last_elements_list(mapped_items)

        pad_mode = "left" if random.random() < 0.5 else "right"
        mapped_items = pad_list(mapped_items, history_size=self.history_size, mode=pad_mode)
        mask_items = pad_list(mask_items, history_size=self.history_size, mode=pad_mode)
        mask_items = torch.tensor(mask_items, dtype=torch.long)
        mapped_items = torch.tensor(mapped_items, dtype=torch.long)

        return mask_items, mapped_items