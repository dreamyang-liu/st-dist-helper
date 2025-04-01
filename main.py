from argparse import ArgumentParser
import time
import boto3
import json
import subprocess
import logging
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--instance-type", type=str, default="t3.large")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--region", type=str, default="us-west-2")
    parser.add_argument("--s3-bucket", type=str, required=True)
    parser.add_argument("--train-image-uri", type=str, required=True)
    parser.add_argument("--role-arn", type=str, required=True)
    parser.add_argument("--local-code-path", type=str, required=True)
    parser.add_argument("--entrypoint", type=str, required=True)
    parser.add_argument("--train-args", type=str, required=False)
    parser.add_argument("--input-data", type=str, required=False)
    parser.add_argument("--output-data", type=str, required=False)
    parser.add_argument("--env-setup-command", type=str, required=False)
    parser.add_argument("--job-name-prefix", type=str, default="my-training-job")
    return parser.parse_args()

def generate_job_name(train_args, job_name_prefix):
    brt = boto3.client('bedrock-runtime')
    model_id = "anthropic.claude-3-5-haiku-20241022-v1:0"
    prompt = f"Given the training arguments {train_args} and training job name prefix {job_name_prefix}, give a concise and descriptive training job name to highlight the key aspects of the training job in 50 characters or less. You may want to shorten the description, like batch size to bs, epochs to ep, etc. Do not include any additional text or explanations, only output the result"

    try:
        native_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "temperature": 0.5,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
        }
        request = json.dumps(native_request)

        response = brt.invoke_model(modelId=model_id, body=request)
        model_response = json.loads(response["body"].read())
        return model_response["content"][0]['text'].strip()

    except (ClientError, Exception) as e:
        logger.error(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        return ""
import os

def create_entrypoint_script(args):
    script_content = f"""#!/bin/bash
        set -e
        cd /opt/ml/input/data/code
        eval "$(/root/miniconda3/bin/conda shell.bash hook)"
        {args.env_setup_command or ''}
        {args.entrypoint} {args.train_args or ''}
        """
    script_filename = 'entrypoint.sh'
    script_path = os.path.join(args.local_code_path, script_filename)

    with open(script_path, 'w') as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)  # Make the script executable

    return script_filename

def sync_code_to_s3(local_path, s3_bucket):
    code_uri = f"s3://{s3_bucket}/code/{local_path.split('/')[-1]}"
    sync_command = f"aws s3 sync {local_path} {code_uri} --exclude \".*\" --exclude \"*/.*\""
    result = subprocess.run(sync_command, shell=True, check=True, capture_output=True, text=True)

    if result.returncode == 0:
        logger.info(f"Successfully synced {local_path} to {code_uri}")
        return code_uri
    else:
        logger.error(f"Error syncing to S3: {result.stderr}")
        exit(1)

def create_training_job(args, code_uri):
    sagemaker = boto3.client('sagemaker')

    GO_TO_CODE_DIR_AND_ACTIVATE_CONDA = "cd /opt/ml/input/data/code && eval \"$(/root/miniconda3/bin/conda shell.bash hook)\""

    response = sagemaker.create_training_job(
        TrainingJobName=f"{args.job_name_prefix}-{int(time.time())}",
        AlgorithmSpecification={
            'TrainingImage': args.train_image_uri,
            'TrainingInputMode': 'File',
            'ContainerEntrypoint': ['bash'],
            'ContainerArguments': ['-c', f"{GO_TO_CODE_DIR_AND_ACTIVATE_CONDA} && {args.env_setup_command} && {args.entrypoint} {args.train_args}"]
        },
        RoleArn=args.role_arn,
        InputDataConfig=[
            {
                'ChannelName': 'code',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataDistributionType': 'FullyReplicated',
                        'S3DataType': 'S3Prefix',
                        'S3Uri': code_uri
                    }
                }
            }
        ],
        ResourceConfig={
            'InstanceType': args.instance_type,
            'InstanceCount': args.instance_count,
            'VolumeSizeInGB': 10
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': 86400
        },
        OutputDataConfig={
            'S3OutputPath': f"s3://{args.s3_bucket}/output"
        },
        EnableManagedSpotTraining=False,
        CheckpointConfig={
            'S3Uri': f"s3://{args.s3_bucket}/checkpoints"
        },
        Environment={},
        RetryStrategy={
            'MaximumRetryAttempts': 1
        },
        Tags=[]
    )
    return response

def main():
    args = parse_arguments()
    args.job_name_prefix = generate_job_name(args.train_args, args.job_name_prefix)
    code_uri = sync_code_to_s3(args.local_code_path, args.s3_bucket)
    response = create_training_job(args, code_uri)
    logger.info(f"Training job created: {response}")

if __name__ == "__main__":
    main()
