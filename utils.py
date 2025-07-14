
import torch
from torch.utils.data import Dataset
import json
import os
import pandas as pd

def preprocess(dataset_path, dataset_saved_path,task):

    if os.path.isdir(dataset_saved_path):
        if task == 'non-overlap':
            return pd.read_csv(os.path.join(dataset_saved_path,"processed_data.csv"))

    try:
        if 'douban' in dataset_path:
            dataset = pd.read_csv(dataset_path,sep='\t')
            max_seq_length = 200
        elif 'epinions' in dataset_path:
            dataset = pd.read_csv(dataset_path)
            dataset = dataset.rename(columns={'user': 'UserId', 'item': 'ItemId','time': 'Timestamp'})
            max_seq_length = 200
        else:
            raise BaseException(f"Can't recognize the dataset type for {dataset_path}")
    except BaseException as e:
        print(f"Error: {e}")

    item_stat = dataset['ItemId'].value_counts()
    user_stat = dataset['UserId'].value_counts()

    # filter out cold start items and users
    filtered_item = item_stat[item_stat>10].keys()
    filtered_user = user_stat[user_stat>10].keys()

    filtered_dataset = dataset[dataset['UserId'].isin(filtered_user) & dataset['ItemId'].isin(filtered_item)]
    filtered_dataset = filtered_dataset.sort_values('Timestamp').groupby('UserId').apply(lambda x:x.tail(max_seq_length)).reset_index(drop=True)

    os.mkdir(dataset_saved_path)
    filtered_dataset.to_csv(os.path.join(dataset_saved_path,"processed_data.csv"))

    return filtered_dataset


def label_split(dataset):
    labels = []
    for user_id, group in dataset.groupby('UserId'):
        if len(group) >= 3:  # Ensure there are at least 3 entries
            test_label = group.iloc[-1]  # Last one for test
            validation_label = group.iloc[-2]  # Second last for validation
            training_label = group.iloc[-3]  # Third last for training

            labels.append({
                'userId': user_id,
                'test_label': test_label['value'],
                'validation_label': validation_label['value'],
                'training_label': training_label['value']
            })
    labels = pd.DataFrame(labels)
    return labels


def load_pretrained_embeddings(parquet_path):
    df = pd.read_parquet(parquet_path)

    num_items = df['ItemId'].max()

    embedding_dim = len(df['item_text_embedding'][0])
    embedding_tensor = torch.zeros((num_items + 1, embedding_dim))

    for index, row in df.iterrows():
        item_id = int(row['ItemId'])
        embedding_tensor[item_id] = torch.tensor(row['item_text_embedding'])
    return embedding_tensor

# Preprocess Script
def load_user_reviews(file_path):
    """Load user reviews from a JSONL file in a cross-platform manner."""
    reviews = []

    # Open the file with robust encoding handling
    with open(file_path, 'r', encoding='utf-8-sig') as fp:  # utf-8-sig handles BOM
        for i, line in enumerate(fp, start=1):
            try:
                # Normalize line endings and strip extra spaces
                normalized_line = line.replace('\r\n', '\n').strip()
                reviews.append(json.loads(normalized_line))  # Parse JSON
            except json.JSONDecodeError as e:
                # Log errors for debugging
                print(f"Error decoding JSON on line {i}: {line.strip()}")
                print(f"JSONDecodeError: {e}")
                continue  # Skip invalid lines

    return reviews


def load_item_metadata(file_path):
    """Load item metadata from a JSONL file in a cross-platform manner."""
    metadata = []

    # Open the file with robust encoding handling
    with open(file_path, 'r', encoding='utf-8-sig') as fp:  # utf-8-sig handles BOM
        for i, line in enumerate(fp, start=1):
            try:
                # Normalize line endings and strip extra spaces
                normalized_line = line.replace('\r\n', '\n').strip()
                metadata.append(json.loads(normalized_line))  # Parse JSON
            except json.JSONDecodeError as e:
                # Log errors for debugging
                print(f"Error decoding JSON on line {i}: {line.strip()}")
                print(f"JSONDecodeError: {e}")
                continue  # Skip invalid lines

    return metadata

def preprocess_data(reviews, metadata):
    """Remove items and users that occurred less than 5 times and filter for the most recent 5 years."""
    # Convert lists of dictionaries to dataframes
    reviews_df = pd.DataFrame(reviews)


    # Filter for the most recent 5 years
    reviews_df['timestamp'] = pd.to_datetime(reviews_df['timestamp'],unit='ms')
    current_year = 2023
    five_years_ago = current_year - 5
    reviews_df = reviews_df[reviews_df['timestamp'].dt.year >= five_years_ago]

    # Filter items and users that occur less than 5 times
    item_counts = reviews_df['parent_asin'].value_counts()
    frequent_items = item_counts[item_counts >= 5].index
    reviews_df = reviews_df[reviews_df['parent_asin'].isin(frequent_items)]
    user_counts = reviews_df['user_id'].value_counts()
    frequent_users = user_counts[user_counts >= 5].index
    reviews_df = reviews_df[reviews_df['user_id'].isin(frequent_users)]



    """Merge user reviews and item metadata to create a complete dataset."""

    reviews_df = reviews_df[['timestamp', 'rating', 'parent_asin', 'user_id', 'verified_purchase']]
    metadata_df = pd.DataFrame(metadata)
    metadata_df = metadata_df[['parent_asin', 'title', 'description']]


    reviews_df.rename(columns={'timestamp': 'Timestamp', 'parent_asin': 'ItemId', 'user_id': 'UserId'}, inplace=True)
    metadata_df.rename(columns={'parent_asin': 'ItemId'}, inplace=True)
    combined_df = reviews_df.merge(metadata_df, how='left', on='ItemId')
    return combined_df


def save_preprocessed_data(df, output_file):
    """Save the preprocessed dataset to a CSV file."""
    df.to_csv(output_file, index=False)



