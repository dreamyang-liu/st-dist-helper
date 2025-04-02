# SageMaker Distributed Training Helper
This repo contains some helper scripts to simplify the process of running distributed training jobs on SageMaker. 

The script `main.py` will be responsible for:
1. Generate a entrypoint script for multi-node distributed training out of box
2. Uploading the training code to S3
3. Creating a SageMaker training job which will download the training code from S3 and run the training command

*Note: You can set `--auto-job-name` flag to automatically generate a job name based on the training command using Bedrock. This requires you to have access to Bedrock and the access to model `anthropic.claude-3-5-haiku-20241022-v1:0`. This will be helpful when you want to run multiple training jobs and want to keep track of them easily.*

**You can check the generated entrypoint script in the `sm-entrypoint.sh` file which located in your local code path.**

The generated entrypoint script will be responsible for: 
1. Determine the how many nodes and how many GPUs on each node are available in the cluster and set the launch command accordingly
2. Configure the environment variables for distributed training
3. Launch the training script using torchrun


## What you need to do (One time setting)
1. Have your AWS credential set up in local machine
2. Create a S3 bucket to store the training code if you don't have one.
3. Create a SageMaker execution role with SageMakerFullAccess permission if you don't have one. The role need to be able to access the S3 bucket.
4. Prepare a training image in ECR repo. (We have a base image that should be good with most case. You can download it from dockerhub and move it to your ECR repo. `timemagic/training-image:py312-pt26-cuda124-transformers`, it has miniconda installed and one prebuilt environment `py_312_torch_transformers`)

## What you need to do (Every time you want to run a training job)
1. Prepare your training code and put it in a folder, and make sure it is written in a way that it can be run with torchrun
2. Configure the parameters in `submit.sh` and run it using `bash submit.sh`

*Note: You don't need to specify the interpreter, the entrypoint script will take care of it. You just need to provide the entrypoint python file and arguments like `main.py --epochs=10 --batch-size=256`*

*Currently we don't support providing S3 input data for training. You need to download the data in your training code. Let us know if you want the feature.* 

**You can assume the train-command and env-setup-command will be run in the foler where the training code is located.**


## Example
In the repo, we provide an example of launching a distributed training job to train resnet50 on FashionMNIST dataset. You can check the existing `submit.sh`.

```shell
python main.py \
    --instance-type ml.g6.12xlarge \
    --instance-count 4 \
    --region us-west-2 \
    --s3-bucket sagemaker-us-west-2-<account-id> \
    --train-image-uri =<account-id>.dkr.ecr.us-west-2.amazonaws.com/training-base:py312_torch_transformers-v1.1 \
    --role-arn arn:aws:iam::<account-id>:role/service-role/AmazonSageMaker-ExecutionRole-20240229T124003 \
    --local-code-path ./resnet50 \
    --env-setup-command "eval \"$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate py_312_torch_transformers" \
    --train-command 'main.py --epochs=10 --batch-size=256' \
    --auto-job-name \
    --job-name-prefix resnet50-fashion-mnist
```

Note: `eval "$(/root/miniconda3/bin/conda shell.bash hook)"` is used to activate conda in the shell.

You can directly run the `submit.sh` to submit the training job as a quick exercise. Remember to use your own AWS account.


## Plesae let me know if you find any issues or want any feature. 