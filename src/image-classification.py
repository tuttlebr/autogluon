import autogluon.core as ag
from autogluon.vision import ImageClassification as task

filename = ag.download("https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip")
ag.unzip(filename)

filename = ag.download("https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip")
ag.unzip(filename)

test_dataset = task.Dataset("data/test", train=False)

if ag.get_gpu_count() == 0:
    dataset = task.Dataset(name="FashionMNIST")
    test_dataset = task.Dataset(name="FashionMNIST", train=False)

classifier = task.fit(dataset, epochs=5, ngpus_per_trial=1, verbose=False)

print("Top-1 val acc: %.3f" % classifier.results["best_reward"])
