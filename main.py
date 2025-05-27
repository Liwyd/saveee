from neural_recommender import NeuralRecommender


# Training mode
# recommender = NeuralRecommender("music_emotional_features.csv", train_model=True, epochs=200)
# Later... load without training
# recommender = NeuralRecommender("music_emotional_features.csv", train_model=False)

recommender = NeuralRecommender("music_emotional_features.csv", epochs=200)
similar = recommender.recommend("Arta - Jealous ft koorosh")


for s in similar:
    print(s)
