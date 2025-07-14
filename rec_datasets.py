from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import torch.nn as nn


class SteamDataset(Dataset):
    def __init__(self, data, max_seq_length):


        unique_item_ids = data['ItemId'].unique()
        self.item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_item_ids, start=0)}
        data['ItemId'] = data['ItemId'].map(self.item_id_map)
        self.num_items = len(self.item_id_map)

        self.data = data[['UserId', 'ItemId', 'Timestamp']].values
        self.max_seq_length = max_seq_length

        self.user_sequences = self._create_user_sequences()
        print(f"item number: {self.num_items}, user number: {len(self.user_sequences)}")

    def _create_user_sequences(self):
        user_sequences = {}
        for user_id, item_id, timestamp in self.data:
            if user_id not in user_sequences:
                user_sequences[user_id] = []
            user_sequences[user_id].append((item_id, timestamp))

        for user_id in user_sequences:
            user_sequences[user_id] = [x[0] for x in sorted(user_sequences[user_id], key=lambda x: x[1])]

        return list(user_sequences.values())

    def __len__(self):
        return len(self.user_sequences)

    def get_num_items(self):
        return self.num_items

    def __getitem__(self, idx):
        sequence = self.user_sequences[idx]
        padded_sequence = np.zeros(self.max_seq_length, dtype=np.int64)
        seq_length = min(len(sequence), self.max_seq_length)
        padded_sequence[-seq_length:] = sequence[-seq_length:]

        # Train with all but last two items, validate with second last, test with last
        train_seq = padded_sequence[:-2]
        val_item = padded_sequence[-2]
        test_item = padded_sequence[-1]

        return torch.tensor(train_seq), torch.tensor(val_item), torch.tensor(test_item)


# Custom dataset for Amazon Sequential Recommendation
class AmazonUserSequencesDataset(Dataset):
    def __init__(self, data, max_seq_length):
        """
        Args:
            csv_file (string): Path to the CSV file with reviews and metadata.
            max_seq_length (int): Maximum sequence length for user interactions.
        """
        self.data_frame = data
        # Re-map ItemId and UserId values to be within valid ranges
        unique_item_ids = self.data_frame['ItemId'].unique()
        self.item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_item_ids, start=0)}
        self.data_frame['ItemId'] = self.data_frame['ItemId'].map(self.item_id_map)

        unique_user_ids = self.data_frame['UserId'].unique()
        self.user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_user_ids, start=0)}
        self.data_frame['UserId'] = self.data_frame['UserId'].map(self.user_id_map)

        # Re-map ItemId values to be within the valid range (1, num_items)
        unique_item_ids = self.data_frame['ItemId'].unique()
        self.item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_item_ids, start=1)}  # Start from 1
        self.data_frame['ItemId'] = self.data_frame['ItemId'].map(self.item_id_map)
        self.max_seq_length = max_seq_length

        # Preprocess data to create user sequences
        self.user_sequences = self._create_user_sequences()
        self.avg_seq_length = sum(len(seq) for seq in self.user_sequences) / len(self.user_sequences) if len(
            self.user_sequences) > 0 else 0
        print(f"Average sequence length: {self.avg_seq_length}")
        self.num_items = len(self.data_frame['ItemId'].unique())
        print(f"Number of items: {self.num_items}, Number of users: {len(self.user_sequences)}")

    def _create_user_sequences(self):
        user_sequences = {}
        for _, row in self.data_frame.iterrows():
            UserId, item_id, timestamp = row['UserId'], row['ItemId'], row['Timestamp']
            if UserId not in user_sequences:
                user_sequences[UserId] = []
            user_sequences[UserId].append((item_id, timestamp))

        # Sort interactions by timestamp and extract item IDs
        for UserId in user_sequences:
            user_sequences[UserId] = [x[0] for x in sorted(user_sequences[UserId], key=lambda x: x[1])]

        return list(user_sequences.values())

    def __len__(self):
        return len(self.user_sequences)

    def get_num_items(self):
        return self.num_items

    def __getitem__(self, idx):
        sequence = self.user_sequences[idx]
        padded_sequence = np.full(self.max_seq_length, 0, dtype=np.int64)
        seq_length = min(len(sequence), self.max_seq_length)
        padded_sequence[-seq_length:] = sequence[-seq_length:]

        # Train with all but last two items, validate with second last, test with last
        train_seq = padded_sequence[:-2]
        val_item = padded_sequence[-2]
        test_item = padded_sequence[-1]

        return torch.tensor(train_seq), torch.tensor(val_item), torch.tensor(test_item)


