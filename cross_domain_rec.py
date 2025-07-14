from utils import *
from models.model_trainer import *
from models.sasrec_sem import *
from models.gru4rec_sem import *
from models.bert4rec_sem import *
from rec_datasets import *
import torch.optim as optim
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_units', type=int, default=256, help='Dimension of the embeddings and hidden layer')
    parser.add_argument('--max_len', type=int, default=50, help='Maximum length of the user interaction sequence')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads (only for SASRec)')
    parser.add_argument('--model_name', type=str, default='gru4rec', help='Model name (sasrec/gru4rec/bert4rec)')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--semantic_embedding', action='store_true', help='Perform semantic embedding')
    parser.add_argument('--top_k', type=int, default=10, help='Top K for evaluation metrics')
    parser.add_argument('--dataset_name', type=str, default='amazon_video_games', help='amazon_musical_instruments/amazon_industrial_and_scientific/amazon_video_games/steam')
    # parser.add_argument('--target_dataset_name', type=str, default='amazon_musical_instruments',
    #                     help='amazon_musical_instruments/amazon_industrial_and_scientific/amazon_video_games')
    parser.add_argument('--model_path', type=str, default='./saved_ckpts/', help='Path to save/load the model')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--force_training', action='store_true', help='do not load dict and train from scratch')
    parser.add_argument('--grid_search', action='store_true', help='Perform grid search for parameter selection')
    args = parser.parse_args()

    def initialize_dataset(dataset_name):
        task = 'non-overlap' # user-overlap, non-overlap

        pretrained_path = f"./data/{dataset_name}/{dataset_name}_embedding_llama3.parquet"
        dataset_saved_path = f'./data/{dataset_name}'

        pretrained_item_embeddings = load_pretrained_embeddings(pretrained_path)


        data = pd.read_csv(os.path.join(dataset_saved_path, "processed_data.csv"))
        dataset = AmazonUserSequencesDataset(data, args.max_len)

        return dataset, pretrained_item_embeddings


    def initialize_model(args, num_items, pretrained_item_embeddings, device):
        if args.model_name.lower() == 'sasrec':
            return SASRec(
                hidden_units=args.hidden_units,
                max_seq_length=args.max_len,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                dropout_rate=args.dropout_rate,
                pretrained_item_embeddings=pretrained_item_embeddings
            ).to(device)
        elif args.model_name.lower() == 'gru4rec':
            return GRU4Rec(
                hidden_units=args.hidden_units,
                num_layers=args.num_layers,
                dropout_rate=args.dropout_rate,
                pretrained_item_embeddings=pretrained_item_embeddings
            ).to(device)
        elif args.model_name.lower() == 'bert4rec':
            return BERT4Rec(
                hidden_units=args.hidden_units,
                max_seq_length=args.max_len,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                dropout_rate=args.dropout_rate,
                pretrained_item_embeddings=pretrained_item_embeddings
            ).to(device)
        else:
            raise ValueError(
                f"Unsupported model name: {args.model_name}. Choose from ['sasrec', 'gru4rec', 'bert4rec']")


    dataset,pretrain_item_embedding = initialize_dataset(args.dataset_name)
    num_items = dataset.get_num_items()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.semantic_embedding:
        model_save_path = os.path.join(args.model_path, f"{args.model_name}_{args.dataset_name}_semantic.pth")
    else:
        model_save_path = os.path.join(args.model_path, f"{args.model_name}_{args.dataset_name}.pth")

    model = initialize_model(args, num_items, pretrain_item_embedding, device)


    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if os.path.exists(model_save_path) and not args.force_training:
        print(f"Loading model from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path))
        print(f"Loaded model successfully")
    else:
        print("Training from scratch")
        train_model(model, dataloader, optimizer, args.num_epochs,num_items, args.early_stop_patience, model_save_path, device)

    evaluate_model_with_neg_sampling(model, dataloader, [1, 5, 10, 20], num_items, device)

    # cross-domain evaluation
    aux_dataset_list = ['amazon_musical_instruments', 'amazon_industrial_and_scientific', 'amazon_video_games']
    aux_dataset_list = set(aux_dataset_list)
    aux_dataset_list.remove(args.dataset_name)
    aux_dataset_list = list(aux_dataset_list)

    for target_dataset_name in aux_dataset_list:
        print(f"Testing results of {target_dataset_name} based on {args.model_name} and {args.dataset_name}")
        target_dataset, target_pretrained_item_embeddings = initialize_dataset(target_dataset_name)
        model.load_new_pretrain_embeddings(target_pretrained_item_embeddings)
        test_dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)
        test_num_items = target_dataset.get_num_items()
        evaluate_model_with_neg_sampling(model, test_dataloader, [1, 5, 10, 20], test_num_items, device)
