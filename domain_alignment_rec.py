
import os
import argparse
import logging
import torch
import torch.optim as optim

from torch.utils.data import DataLoader

# Model & utility imports
from utils import *
from models.bert4rec_sem import BERT4Rec
from models.sasrec_sem import SASRec
from models.gru_recg import GRU4RecWithDomainAlignment
from models.model_trainer import train_model_with_alignment, evaluate_model_with_neg_sampling
from rec_datasets import AmazonUserSequencesDataset, SteamDataset, preprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_units', type=int, default=256)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='gru4rec')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--semantic_embedding', action='store_true')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--dataset_name', type=str, default='amazon_musical_instruments')
    parser.add_argument('--model_path', type=str, default='./saved_ckpts/')
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--force_training', action='store_true')
    parser.add_argument('--grid_search', action='store_true')
    return parser.parse_args()


def load_pretrained_embeddings_from_dataset(dataset_name):
    path = f"./data/{dataset_name}/{dataset_name}_embedding_llama3.parquet"
    return load_pretrained_embeddings(path)


def initialize_dataset(dataset_name, max_len):
    task = 'non-overlap'
    dataset_path = f'./data/source_files/{dataset_name}.csv'
    dataset_saved_path = f'./data/{dataset_name}'

    if 'amazon' in dataset_name:
        data = pd.read_csv(os.path.join(dataset_saved_path, "processed_data.csv"))
        return AmazonUserSequencesDataset(data, max_len)
    else:
        preprocessed = preprocess(dataset_path, dataset_saved_path, task)
        return SteamDataset(preprocessed, max_len)


def initialize_model(args, num_items, pretrained_item_embeddings, device):
    model_name = args.model_name.lower()
    if model_name == 'sasrec':
        return SASRec(
            hidden_units=args.hidden_units,
            max_seq_length=args.max_len,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate,
            pretrained_item_embeddings=pretrained_item_embeddings
        ).to(device)
    elif model_name == 'gru4rec':
        return GRU4RecWithDomainAlignment(
            hidden_units=args.hidden_units,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate,
            pretrained_item_embeddings=pretrained_item_embeddings
        ).to(device)
    elif model_name == 'bert4rec':
        return BERT4Rec(
            hidden_units=args.hidden_units,
            max_seq_length=args.max_len,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate,
            pretrained_item_embeddings=pretrained_item_embeddings
        ).to(device)
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")


def sample_auxiliary_domains(device, dataset_name, num_samples=1024):
    aux_datasets = {'amazon_musical_instruments', 'amazon_industrial_and_scientific', 'amazon_video_games', 'steam'}
    aux_datasets.discard(dataset_name)

    sampled_embeddings, sampled_domains = [], []

    for domain_id, aux_name in enumerate(sorted(aux_datasets)):
        aux_embeddings = load_pretrained_embeddings_from_dataset(aux_name)
        indices = torch.randperm(aux_embeddings.size(0))[:num_samples]
        sampled_embeddings.append(aux_embeddings[indices])
        sampled_domains.append(torch.full((num_samples,), domain_id, dtype=torch.long))

    return torch.cat(sampled_embeddings).to(device), torch.cat(sampled_domains).to(device), sorted(aux_datasets)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")

    dataset = initialize_dataset(args.dataset_name, args.max_len)
    pretrained_embeddings = load_pretrained_embeddings_from_dataset(args.dataset_name)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    num_items = dataset.get_num_items()

    model = initialize_model(args, num_items, pretrained_embeddings, device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model_save_path = os.path.join(args.model_path, f"{args.model_name}_{args.dataset_name}_irm.pth")

    sampled_embeddings, sampled_domains, aux_dataset_list = sample_auxiliary_domains(device, args.dataset_name)

    if os.path.exists(model_save_path) and not args.force_training:
        logger.info(f"Loading model from: {model_save_path}")
        model.load_state_dict(torch.load(model_save_path))
        logger.info("Model loaded successfully.")
    else:
        logger.info("Training model from scratch...")
        train_model_with_alignment(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            num_epochs=args.num_epochs,
            num_items=num_items,
            num_domains=len(aux_dataset_list),
            sampled_domains=sampled_domains,
            sampled_embeddings=sampled_embeddings,
            alpha=0.01,
            early_stop_patience=args.early_stop_patience,
            model_save_path=model_save_path,
            device=device
        )

    logger.info("Evaluating on source domain...")
    evaluate_model_with_neg_sampling(model, dataloader, [5, 10, 20], num_items, device)

    for target_name in aux_dataset_list:
        logger.info(f"Evaluating transfer to target domain: {target_name}")
        target_dataset = initialize_dataset(target_name, args.max_len)
        target_embeddings = load_pretrained_embeddings_from_dataset(target_name)
        model.load_new_pretrain_embeddings(target_embeddings)

        test_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)
        test_num_items = target_dataset.get_num_items()

        evaluate_model_with_neg_sampling(model, test_loader, [5, 10, 20], test_num_items, device)


if __name__ == '__main__':
    main()
