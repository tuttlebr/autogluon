# [AutoGluon](https://auto.gluon.ai/stable/index.html)

Autogluon is the work of many people (At Amazon maybe?) building from the work of Nick Erickson. Check out the paper [AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data](https://arxiv.org/pdf/2003.06505). This repository exists for those who use Docker and want to give it a few test cases in isolation.

## AutoGluon: AutoML for Text, Image, and Tabular Data

AutoGluon enables easy-to-use and easy-to-extend AutoML with a focus on automated stack ensembling, deep learning, and real-world applications spanning text, image, and tabular data. Intended for both ML beginners and experts, AutoGluon enables you to:

- Quickly prototype deep learning and classical ML solutions for your raw data with a few lines of code.
- Automatically utilize state-of-the-art techniques (where appropriate) without expert knowledge.
- Leverage automatic hyperparameter tuning, model selection/ensembling, architecture search, and data processing.
- Easily improve/tune your bespoke models and data pipelines, or customize AutoGluon for your use-case.

## Options

Modify the .env file to specify something to run within the Autogluon env. Expects a system with a GPU which may be accessed by Docker.

```shell
DOCKER_BUILTKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker-compose build
docker-compose up
```
