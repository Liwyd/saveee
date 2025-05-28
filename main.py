from recommender import NeuralRecommender  # فرض می‌گیریم کد اصلی توی فایل recommender.py هست

def main():
    lmdb_path = "music_embeddings.lmdb"
    model_path = "recommender.pt"
    recommender = NeuralRecommender(lmdb_path, model_path=model_path, epochs=200)

    while True:
        track_name = input("Enter track name (or 'exit' to quit): ").strip()
        if track_name.lower() == "exit":
            break

        try:
            recommendations = recommender.recommend(track_name, top_k=5)
            print(f"Top similar tracks to '{track_name}':")
            for idx, name in recommendations:
                print(f"  - [{idx}] {name}")
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
