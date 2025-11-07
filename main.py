import argparse
import torch
from tokenizer import byte_pair_tokenizer
from model import model_predictor
from data import batch_generator
from train import train
import os 

def main():
    parser = argparse.ArgumentParser(description="Train a small LLM on Friends transcripts")
    # --- User related ---
    parser.add_argument("--train",action="store_true",help="Train from scratch instead of loading a checkpoint")
    parser.add_argument("--ckpt_save_path", type=str, default="llm_friends.pt", help="Path to checkpoint")
    parser.add_argument("--predict", type=str, default=" ", help="first token to begin prediction")


    # --- Tokenizer-related arguments ---
    parser.add_argument("--num_merges", type=int, default=10, help="Number of BPE merges to perform")
    parser.add_argument("--tokenizer_fraction", type=float, default=0.1, help="Fraction of corpus to train tokenizer on")
    parser.add_argument("--tokenizer_path", type=str, default="vocabulary.json", help="Path to tokenizer JSON file")

    # --- Model-related arguments ---
    parser.add_argument("--context_size", type=int, default=20, help="Context window size for the model")
    parser.add_argument("--nb_layers", type=int, default=6, help="Number of transformer layers")

    # --- Training-related arguments ---
    parser.add_argument("--training_iterations", type=int, default=5000, help="Number of training iterations")
    parser.add_argument("--eval_interval", type=int, default=50, help="How often to evaluate validation loss")

    # --- Data-related arguments ---
    parser.add_argument("--data_path", type=str, default="Friends_Transcript/Friends_Transcript.txt", help="Path to text corpus")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Train/val split ratio")

    args = parser.parse_args()

    # --- Device setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Everything will be running on {device}")

    # *********************--- Training the tokenizer ---*********************
    tokenizer = byte_pair_tokenizer(
        fraction=args.tokenizer_fraction,
        num_merges=args.num_merges,
        tokenizer_path=args.tokenizer_path
    )
    vocab_size = len(tokenizer.vocab)

    # *********************--- Model setup ---*********************
    model = model_predictor(nb_layers=args.nb_layers, vocabulary_length=vocab_size)
    model.to(device)

    with open(args.data_path, encoding="utf-8") as f:
        corpus = f.read()

    data_idx = tokenizer.encode_using_regex(corpus)
    batch_gen = batch_generator(
        data_idx,
        split_ratio=args.split_ratio,
        context_size=args.context_size,
        device=device
    )

    # *********************--- Training the model ---*********************
    if args.train:
        # If a checkpoint exists, resume from it
        if os.path.exists(args.ckpt_save_path):
            print(f"Checkpoint found at {args.ckpt_save_path}. Resuming training...")
            model.load_state_dict(torch.load(args.ckpt_save_path, map_location=device))
        else:
            print("No checkpoint found. Starting training from scratch.")

        # Start or resume training
        train(
            model=model,
            batch_generator=batch_gen,
            iterations=args.training_iterations,
            eval_interval=args.eval_interval
        )

    else:
        # If not training, load existing model and run inference/evaluation
        if not os.path.exists(args.ckpt_save_path):
            raise FileNotFoundError(
                f"No checkpoint found at {args.ckpt_save_path}. "
                "Use --train to train a new model."
            )

        print(f"Loading trained model from {args.ckpt_save_path} for inference...")
        model.load_state_dict(torch.load(args.ckpt_save_path, map_location=device))
        model.eval()
        print("performing the prediction")
        print(tokenizer.decode(model.predict(
            idx= torch.tensor(tokenizer.encode(args.predict)).unsqueeze(0).to(device) 
            , num_predictions=2000)[0].tolist()))


if __name__ == "__main__":
    main()



