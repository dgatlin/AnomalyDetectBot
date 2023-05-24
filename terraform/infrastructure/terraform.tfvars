project_name = "anomaly-ml-pipeline-terraform"
region = "us-east-1"

## Change instance types amd volume size for SageMaker
training_instance_type = "ml.m5.xlarge"
inference_instance_type = "ml.c5.large"
volume_size_sagemaker = 5

## Should not be changed with the current folder structure
handler_path  = "../../lambda_function"
handler       = "config_lambda.lambda_handler"
