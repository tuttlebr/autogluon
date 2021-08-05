from autogluon.core.utils.loaders.load_pd import load

train_data = load(
    "https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet"
)
test_data = load(
    "https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet"
)
subsample_size = (
    1000  # subsample data for faster demo, try setting this to larger values
)
train_data = train_data.sample(n=subsample_size, random_state=0)
train_data.head(10)

from autogluon.text import TextPredictor

predictor = TextPredictor(label="label", eval_metric="acc", path="./ag_sst")
predictor.fit(train_data, time_limit=60)
test_score = predictor.evaluate(test_data)
print("Accuracy = {:.2f}%".format(test_score * 100))

test_score = predictor.evaluate(test_data, metrics=["acc", "f1"])
print(test_score)

sentence1 = "it's a charming and often affecting journey."
sentence2 = "It's slow, very, very, very slow."
predictions = predictor.predict({"sentence": [sentence1, sentence2]})
print('"Sentence":', sentence1, '"Predicted Sentiment":', predictions[0])
print('"Sentence":', sentence2, '"Predicted Sentiment":', predictions[1])

probs = predictor.predict_proba({"sentence": [sentence1, sentence2]})
print('"Sentence":', sentence1, '"Predicted Class-Probabilities":', probs[0])
print('"Sentence":', sentence2, '"Predicted Class-Probabilities":', probs[1])

embeddings = predictor.extract_embedding(test_data)
print(embeddings)
