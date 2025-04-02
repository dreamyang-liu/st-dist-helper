from argparse import ArgumentParser
import time
import boto3
import json
import subprocess
import logging
import os
from jinja2 import Template
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--instance-type", type=str, default="t3.large", help="EC2 instance type for training")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of instances to use for training")
    parser.add_argument("--region", type=str, default="us-west-2", help="AWS region to run the training job")
    parser.add_argument("--s3-bucket", type=str, required=True, help="S3 bucket used for storing training code and output")
    parser.add_argument("--max-runtime", type=int, default=86400, help="Maximum runtime for the training job in seconds")
    parser.add_argument("--train-image-uri", type=str, required=True, help="URI of the Docker image for training")
    parser.add_argument("--role-arn", type=str, required=True, help="ARN of the IAM role for SageMaker to assume")
    parser.add_argument("--local-code-path", type=str, required=True, help="Local path to the training code")
    parser.add_argument("--train-command", type=str, required=False, help="Command to run for training")
    parser.add_argument("--env-setup-command", type=str, required=False, help="Command to set up the environment, e.g., 'pip install -r requirements.txt'")
    parser.add_argument("--auto-job-name", action="store_true", help="Automatically generate a job name based on the training command using Bedrock")
    parser.add_argument("--job-name-prefix", type=str, default="my-training-job", help="Prefix for the training job name")
    return parser.parse_args()

def generate_job_name(train_command, job_name_prefix):
    brt = boto3.client('bedrock-runtime')
    model_id = "anthropic.claude-3-5-haiku-20241022-v1:0"
    prompt = f"Given the training command {train_command} and training job name prefix {job_name_prefix}, give a concise and descriptive training job name to highlight the key aspects of the training job in 50 characters or less. You may want to shorten the description, like batch size to bs, epochs to ep, etc. Do not include any additional text or explanations, only output the result"

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

def render_template(args, **kwargs):
    with open("entrypoint.jinja", 'r') as f:
        template_content = f.read()
    
    template = Template(template_content)
    rendered_content = template.render(**kwargs)
    
    with open(os.path.join(args.local_code_path, "sm-entrypoint.sh"), 'w') as f:
        f.write(rendered_content)
    os.chmod(os.path.join(args.local_code_path, "sm-entrypoint.sh"), 0o755)

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
    response = sagemaker.create_training_job(
        TrainingJobName=f"{args.job_name_prefix}-{int(time.time())}",
        AlgorithmSpecification={
            'TrainingImage': args.train_image_uri,
            'TrainingInputMode': 'File',
            'ContainerEntrypoint': ['bash'],
            'ContainerArguments': ['-c', 'bash /opt/ml/input/data/code/sm-entrypoint.sh']
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
            'MaxRuntimeInSeconds': args.max_runtime
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
    if args.auto_job_name:
        args.job_name_prefix = generate_job_name(args.train_command, args.job_name_prefix)
    render_template(args, environment_setup_command=args.env_setup_command,
                    entrypoint_command=args.train_command,
                    training_mode='multi-node' if args.instance_count > 1 else 'single-node',
                    trim_blocks=True,
                    lstrip_blocks=True)
    code_uri = sync_code_to_s3(args.local_code_path, args.s3_bucket)
    response = create_training_job(args, code_uri)
    logger.info(f"Training job created: {response}")

if __name__ == "__main__":
    main()
