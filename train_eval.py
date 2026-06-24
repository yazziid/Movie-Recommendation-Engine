import os
import argparse
import pandas as pd
import numpy as np
import torch
import pickle
import faiss

from model_prep import train_df, test_df, TwoTowerDataset, index_mapping, top_k_actor_per_movie, X_item, movies_df
from TwoTowerArchitecture import TwoTowerWithItemFeatures, train_model
from two_tower_metrics import calculate_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Two-Tower Recommendation Model")
    parser.add_argument("--model_name", type=str, default="v1", help="Name of the model version folder")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--emb_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=264, help="Hidden layer dimension")
    parser.add_argument("--batch_size", type=int, default=4096, help="Training batch size")
    args = parser.parse_args()

    # Versioning & Metrics Directories
    model_dir = f"models/{args.model_name}"
    metrics_dir = os.path.join(model_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    print(f"--- Starting Training Run: {args.model_name} ---")

    n_users, n_items, idx2movieid, userid2idx, movieid2idx = index_mapping(train_df)
    A_item, n_actors, K = top_k_actor_per_movie()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    implicit_train = train_df[["userid","movielensid"]].drop_duplicates()
    train_dataset = TwoTowerDataset(implicit_train, userid2idx, movieid2idx)

    model = TwoTowerWithItemFeatures(
        n_users=n_users, n_items=n_items, item_feat_dim=X_item.shape[1],
        n_actors=n_actors, K=K, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim,
        dropout=0.1, normalize=True
    )
    
    print(f"Training on {device} for {args.epochs} epochs...")
    epoch_history = train_model(
        model=model, train_dataset=train_dataset, X_item=X_item, A_item=A_item,
        device=device, epochs=args.epochs, batch_size=args.batch_size
    )

    epoch_df = pd.DataFrame(epoch_history)
    epoch_df.to_csv(f"{metrics_dir}/epoch_metrics.csv", index=False)
    print(f"Epoch loss history saved to {metrics_dir}/epoch_metrics.csv")

    # Evaluate Model
    print("Evaluating model against holdout test set...")
    model.eval()
    X_item_dev = X_item.to(device)
    A_item_dev = A_item.to(device)

    with torch.no_grad():
        item_vectors = model.compute_all_item_vectors(X_item_dev, A_item_dev)

    global_item_counts = train_df['movielensid'].value_counts().to_dict()
    total_train_interactions = len(train_df)

    eval_metrics = calculate_metrics(
        model, test_df, item_vectors, userid2idx, movieid2idx, device, 
        global_item_counts, total_train_interactions, topk=10
    )

    eval_metrics["Hyper_Epochs"] = args.epochs
    eval_metrics["Hyper_Batch_Size"] = args.batch_size
    eval_metrics["Hyper_Emb_Dim"] = args.emb_dim
    eval_metrics["Hyper_Hidden_Dim"] = args.hidden_dim

    print("Final Evaluation Metrics:")
    for k, v in eval_metrics.items():
        print(f" - {k}: {v:.4f}")

    eval_df = pd.DataFrame([eval_metrics])
    eval_df.to_csv(f"{metrics_dir}/evaluation_metrics.csv", index=False)
    print(f"Final evaluation metrics saved to {metrics_dir}/evaluation_metrics.csv")

    print(f"Saving artifacts to {model_dir}/...")
    torch.save(model.state_dict(), f"{model_dir}/two_tower_model.pth")

    model_artifacts = {
        "userid2idx": userid2idx, "movieid2idx": movieid2idx, "idx2movieid": idx2movieid,
        "item_vectors": item_vectors.cpu(), "n_actors": n_actors, "K": K,
        "emb_dim": args.emb_dim, "hidden_dim": args.hidden_dim
    }
    with open(f"{model_dir}/model_artifacts.pkl", "wb") as f:
        pickle.dump(model_artifacts, f)

    item_vectors_np = item_vectors.cpu().numpy().astype('float32')
    faiss_index = faiss.IndexFlatIP(item_vectors_np.shape[1])
    faiss_index.add(item_vectors_np)
    faiss.write_index(faiss_index, f"{model_dir}/movie_vectors.index")

if __name__ == "__main__":
    main()