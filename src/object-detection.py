import autogluon.core as ag
from autogluon.vision import ImagePredictor

from logging import info, basicConfig

basicConfig(format="%(asctime)s %(message)s")

train_dataset, _, test_dataset = ImagePredictor.Dataset.from_folders(
    "https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip"
)
info(train_dataset)

predictor = ImagePredictor()

predictor.fit(train_dataset, hyperparameters={"epochs": 2})
fit_result = predictor.fit_summary()
info(
    "Top-1 train acc: %.3f, val acc: %.3f"
    % (fit_result["train_acc"], fit_result["valid_acc"])
)

image_path = test_dataset.iloc[0]["image"]
result = predictor.predict(image_path)
info(result)

proba = predictor.predict_proba(image_path)
info(proba)

bulk_result = predictor.predict(test_dataset)
info(bulk_result)
