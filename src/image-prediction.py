import autogluon.core as ag
from autogluon.vision import ImagePredictor

train_dataset, _, test_dataset = ImagePredictor.Dataset.from_folders(
    "https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip"
)
print(train_dataset)

predictor = ImagePredictor()
# since the original dataset does not provide validation split, the `fit` function splits it randomly with 90/10 ratio
predictor.fit(
    train_dataset, hyperparameters={"epochs": 2}
)  # you can trust the default config, we reduce the # epoch to save some build time

fit_result = predictor.fit_summary()
print(
    "Top-1 train acc: %.3f, val acc: %.3f"
    % (fit_result["train_acc"], fit_result["valid_acc"])
)

image_path = test_dataset.iloc[0]["image"]
result = predictor.predict(image_path)
print(result)

proba = predictor.predict_proba(image_path)
print(proba)

bulk_result = predictor.predict(test_dataset)
print(bulk_result)
