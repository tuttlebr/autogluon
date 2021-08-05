import autogluon.core as ag
from autogluon.vision import ObjectDetector

url = "https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip"
dataset_train = ObjectDetector.Dataset.from_voc(url, splits="trainval")

time_limit = 60 * 30  # at most 0.5 hour
detector = ObjectDetector()
hyperparameters = {"epochs": 5, "batch_size": 8}
detector.fit(
    dataset_train,
    time_limit=time_limit,
    hyperparameters=hyperparameters,
)

dataset_test = ObjectDetector.Dataset.from_voc(url, splits="test")

test_map = detector.evaluate(dataset_test)
print("mAP on test dataset: {}".format(test_map[1][-1]))

image_path = dataset_test.iloc[0]["image"]
result = detector.predict(image_path)
print(result)

bulk_result = detector.predict(dataset_test)
print(bulk_result)

savefile = "detector.ag"
detector.save(savefile)
new_detector = ObjectDetector.load(savefile)
