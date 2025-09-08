"""
AWS Cloud Integration for NBA Props Analyzer
Handles model deployment, data storage, and real-time processing
"""
import boto3
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from botocore.exceptions import ClientError, NoCredentialsError
import logging
from typing import Dict, List, Any
import asyncio
import aioboto3

class AWSIntegration:
    def __init__(self, region='us-east-1'):
        self.region = region
        self.setup_clients()
        self.setup_logging()
        
    def setup_clients(self):
        """Initialize AWS service clients"""
        try:
            # Core services
            self.s3 = boto3.client('s3', region_name=self.region)
            self.dynamodb = boto3.resource('dynamodb', region_name=self.region)
            self.lambda_client = boto3.client('lambda', region_name=self.region)
            self.sagemaker = boto3.client('sagemaker', region_name=self.region)
            self.cloudwatch = boto3.client('cloudwatch', region_name=self.region)
            
            # Real-time services
            self.kinesis = boto3.client('kinesis', region_name=self.region)
            self.sns = boto3.client('sns', region_name=self.region)
            self.sqs = boto3.client('sqs', region_name=self.region)
            
            # AI/ML services
            self.bedrock = boto3.client('bedrock-runtime', region_name=self.region)
            self.comprehend = boto3.client('comprehend', region_name=self.region)
            
            print("✅ AWS clients initialized successfully")
            
        except NoCredentialsError:
            print("❌ AWS credentials not found. Please configure credentials.")
            raise
        except Exception as e:
            print(f"❌ Error initializing AWS clients: {e}")
            raise
    
    def setup_logging(self):
        """Setup CloudWatch logging"""
        self.logger = logging.getLogger('nba_props_aws')
        self.logger.setLevel(logging.INFO)
        
        # Create CloudWatch handler
        try:
            import watchtower
            handler = watchtower.CloudWatchLogsHandler(
                boto3_client=self.cloudwatch,
                log_group='/nba-props-analyzer/application',
                stream_name=f"instance-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        except ImportError:
            print("Install watchtower for CloudWatch logging: pip install watchtower")

class S3DataManager:
    """Manages data storage and retrieval from S3"""
    
    def __init__(self, aws_integration: AWSIntegration):
        self.s3 = aws_integration.s3
        self.bucket_name = 'nba-props-analyzer-data'
        self.logger = aws_integration.logger
        
        # Create bucket if it doesn't exist
        self.create_bucket_if_not_exists()
    
    def create_bucket_if_not_exists(self):
        """Create S3 bucket if it doesn't exist"""
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                try:
                    self.s3.create_bucket(Bucket=self.bucket_name)
                    
                    # Enable versioning
                    self.s3.put_bucket_versioning(
                        Bucket=self.bucket_name,
                        VersioningConfiguration={'Status': 'Enabled'}
                    )
                    
                    # Set lifecycle policy
                    lifecycle_config = {
                        'Rules': [
                            {
                                'ID': 'DeleteOldVersions',
                                'Status': 'Enabled',
                                'Filter': {'Prefix': 'models/'},
                                'NoncurrentVersionExpiration': {'NoncurrentDays': 30}
                            }
                        ]
                    }
                    self.s3.put_bucket_lifecycle_configuration(
                        Bucket=self.bucket_name,
                        LifecycleConfiguration=lifecycle_config
                    )
                    
                    self.logger.info(f"Created S3 bucket: {self.bucket_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error creating bucket: {e}")
                    raise
    
    def upload_model(self, model_path: str, model_name: str):
        """Upload ML model to S3"""
        try:
            s3_key = f"models/{model_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
            self.s3.upload_file(
                model_path, 
                self.bucket_name, 
                s3_key,
                ExtraArgs={
                    'Metadata': {
                        'model_name': model_name,
                        'upload_time': str(datetime.now()),
                        'version': '1.0'
                    }
                }
            )
            
            self.logger.info(f"Model uploaded: {s3_key}")
            return s3_key
            
        except Exception as e:
            self.logger.error(f"Error uploading model: {e}")
            raise
    
    def download_model(self, s3_key: str, local_path: str):
        """Download ML model from S3"""
        try:
            self.s3.download_file(self.bucket_name, s3_key, local_path)
            self.logger.info(f"Model downloaded: {s3_key} -> {local_path}")
            
        except Exception as e:
            self.logger.error(f"Error downloading model: {e}")
            raise
    
    def store_predictions(self, predictions: Dict[str, Any]):
        """Store predictions in S3"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"predictions/{timestamp}.json"
            
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json.dumps(predictions, default=str),
                ContentType='application/json'
            )
            
            return s3_key
            
        except Exception as e:
            self.logger.error(f"Error storing predictions: {e}")
            raise
    
    def store_training_data(self, data: pd.DataFrame, data_type: str):
        """Store training data in S3"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"training_data/{data_type}/{timestamp}.parquet"
            
            # Convert to parquet for efficient storage
            parquet_buffer = data.to_parquet()
            
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=parquet_buffer,
                ContentType='application/octet-stream'
            )
            
            self.logger.info(f"Training data stored: {s3_key}")
            return s3_key
            
        except Exception as e:
            self.logger.error(f"Error storing training data: {e}")
            raise

class DynamoDBManager:
    """Manages real-time data storage in DynamoDB"""
    
    def __init__(self, aws_integration: AWSIntegration):
        self.dynamodb = aws_integration.dynamodb
        self.logger = aws_integration.logger
        
        # Table names
        self.users_table_name = 'nba-props-users'
        self.predictions_table_name = 'nba-props-predictions'
        self.live_odds_table_name = 'nba-props-live-odds'
        
        self.setup_tables()
    
    def setup_tables(self):
        """Create DynamoDB tables"""
        
        # Users table
        try:
            self.users_table = self.dynamodb.create_table(
                TableName=self.users_table_name,
                KeySchema=[
                    {'AttributeName': 'user_id', 'KeyType': 'HASH'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'user_id', 'AttributeType': 'S'}
                ],
                BillingMode='PAY_PER_REQUEST'
            )
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceInUseException':
                raise
            self.users_table = self.dynamodb.Table(self.users_table_name)
        
        # Predictions table
        try:
            self.predictions_table = self.dynamodb.create_table(
                TableName=self.predictions_table_name,
                KeySchema=[
                    {'AttributeName': 'prediction_id', 'KeyType': 'HASH'},
                    {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'prediction_id', 'AttributeType': 'S'},
                    {'AttributeName': 'timestamp', 'AttributeType': 'S'},
                    {'AttributeName': 'user_id', 'AttributeType': 'S'}
                ],
                GlobalSecondaryIndexes=[
                    {
                        'IndexName': 'user-index',
                        'KeySchema': [
                            {'AttributeName': 'user_id', 'KeyType': 'HASH'},
                            {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                        ],
                        'Projection': {'ProjectionType': 'ALL'}
                    }
                ],
                BillingMode='PAY_PER_REQUEST'
            )
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceInUseException':
                raise
            self.predictions_table = self.dynamodb.Table(self.predictions_table_name)
        
        # Live odds table
        try:
            self.live_odds_table = self.dynamodb.create_table(
                TableName=self.live_odds_table_name,
                KeySchema=[
                    {'AttributeName': 'game_id', 'KeyType': 'HASH'},
                    {'AttributeName': 'prop_type', 'KeyType': 'RANGE'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'game_id', 'AttributeType': 'S'},
                    {'AttributeName': 'prop_type', 'AttributeType': 'S'}
                ],
                BillingMode='PAY_PER_REQUEST',
                StreamSpecification={
                    'StreamEnabled': True,
                    'StreamViewType': 'NEW_AND_OLD_IMAGES'
                }
            )
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceInUseException':
                raise
            self.live_odds_table = self.dynamodb.Table(self.live_odds_table_name)
    
    def store_user_data(self, user_id: str, user_data: Dict[str, Any]):
        """Store user profile data"""
        try:
            user_data['user_id'] = user_id
            user_data['last_updated'] = str(datetime.now())
            
            self.users_table.put_item(Item=user_data)
            self.logger.info(f"User data stored for: {user_id}")
            
        except Exception as e:
            self.logger.error(f"Error storing user data: {e}")
            raise
    
    def get_user_data(self, user_id: str):
        """Retrieve user profile data"""
        try:
            response = self.users_table.get_item(Key={'user_id': user_id})
            return response.get('Item', {})
            
        except Exception as e:
            self.logger.error(f"Error retrieving user data: {e}")
            raise
    
    def store_prediction(self, prediction_data: Dict[str, Any]):
        """Store prediction result"""
        try:
            prediction_data['timestamp'] = str(datetime.now())
            
            self.predictions_table.put_item(Item=prediction_data)
            self.logger.info(f"Prediction stored: {prediction_data.get('prediction_id')}")
            
        except Exception as e:
            self.logger.error(f"Error storing prediction: {e}")
            raise
    
    def update_live_odds(self, game_id: str, prop_type: str, odds_data: Dict[str, Any]):
        """Update live odds data"""
        try:
            odds_data.update({
                'game_id': game_id,
                'prop_type': prop_type,
                'timestamp': str(datetime.now())
            })
            
            self.live_odds_table.put_item(Item=odds_data)
            
        except Exception as e:
            self.logger.error(f"Error updating live odds: {e}")
            raise

class SageMakerDeployment:
    """Handles model deployment on SageMaker"""
    
    def __init__(self, aws_integration: AWSIntegration):
        self.sagemaker = aws_integration.sagemaker
        self.logger = aws_integration.logger
        
    def deploy_model(self, model_s3_path: str, model_name: str):
        """Deploy model to SageMaker endpoint"""
        try:
            # Create model
            model_response = self.sagemaker.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3',
                    'ModelDataUrl': model_s3_path,
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'inference.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
                    }
                },
                ExecutionRoleArn='arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole'
            )
            
            # Create endpoint configuration
            endpoint_config_name = f"{model_name}-config"
            config_response = self.sagemaker.create_endpoint_configuration(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'primary',
                        'ModelName': model_name,
                        'InitialInstanceCount': 1,
                        'InstanceType': 'ml.t2.medium',
                        'InitialVariantWeight': 1.0
                    }
                ]
            )
            
            # Create endpoint
            endpoint_name = f"{model_name}-endpoint"
            endpoint_response = self.sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            
            self.logger.info(f"Model deployment initiated: {endpoint_name}")
            return endpoint_name
            
        except Exception as e:
            self.logger.error(f"Error deploying model: {e}")
            raise
    
    def invoke_endpoint(self, endpoint_name: str, payload: Dict[str, Any]):
        """Invoke SageMaker endpoint for predictions"""
        try:
            runtime = boto3.client('sagemaker-runtime')
            
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            
            result = json.loads(response['Body'].read().decode())
            return result
            
        except Exception as e:
            self.logger.error(f"Error invoking endpoint: {e}")
            raise

class RealTimeProcessor:
    """Handles real-time data processing with Kinesis"""
    
    def __init__(self, aws_integration: AWSIntegration):
        self.kinesis = aws_integration.kinesis
        self.sns = aws_integration.sns
        self.logger = aws_integration.logger
        
        self.stream_name = 'nba-props-live-stream'
        self.topic_arn = None
        
        self.setup_stream()
        self.setup_notifications()
    
    def setup_stream(self):
        """Create Kinesis stream for live data"""
        try:
            self.kinesis.create_stream(
                StreamName=self.stream_name,
                ShardCount=1
            )
            self.logger.info(f"Kinesis stream created: {self.stream_name}")
            
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceInUseException':
                raise
    
    def setup_notifications(self):
        """Setup SNS topic for alerts"""
        try:
            response = self.sns.create_topic(Name='nba-props-alerts')
            self.topic_arn = response['TopicArn']
            
        except Exception as e:
            self.logger.error(f"Error setting up notifications: {e}")
    
    def stream_live_data(self, data: Dict[str, Any]):
        """Stream live data to Kinesis"""
        try:
            self.kinesis.put_record(
                StreamName=self.stream_name,
                Data=json.dumps(data),
                PartitionKey=data.get('game_id', 'default')
            )
            
        except Exception as e:
            self.logger.error(f"Error streaming data: {e}")
            raise
    
    def send_alert(self, message: str, subject: str = "NBA Props Alert"):
        """Send alert via SNS"""
        try:
            if self.topic_arn:
                self.sns.publish(
                    TopicArn=self.topic_arn,
                    Message=message,
                    Subject=subject
                )
                
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")

class CloudMetrics:
    """CloudWatch metrics and monitoring"""
    
    def __init__(self, aws_integration: AWSIntegration):
        self.cloudwatch = aws_integration.cloudwatch
        self.logger = aws_integration.logger
        
        self.namespace = 'NBA-Props-Analyzer'
    
    def put_custom_metric(self, metric_name: str, value: float, unit: str = 'Count'):
        """Put custom metric to CloudWatch"""
        try:
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[
                    {
                        'MetricName': metric_name,
                        'Value': value,
                        'Unit': unit,
                        'Timestamp': datetime.now()
                    }
                ]
            )
            
        except Exception as e:
            self.logger.error(f"Error putting metric: {e}")
    
    def create_alarm(self, alarm_name: str, metric_name: str, threshold: float):
        """Create CloudWatch alarm"""
        try:
            self.cloudwatch.put_metric_alarm(
                AlarmName=alarm_name,
                ComparisonOperator='GreaterThanThreshold',
                EvaluationPeriods=1,
                MetricName=metric_name,
                Namespace=self.namespace,
                Period=300,
                Statistic='Average',
                Threshold=threshold,
                ActionsEnabled=True,
                AlarmActions=[
                    # SNS topic ARN for notifications
                ],
                AlarmDescription=f'Alarm for {metric_name}',
                Unit='Count'
            )
            
        except Exception as e:
            self.logger.error(f"Error creating alarm: {e}")

# Example usage
if __name__ == "__main__":
    # Initialize AWS integration
    aws = AWSIntegration()
    
    # Setup data management
    s3_manager = S3DataManager(aws)
    dynamodb_manager = DynamoDBManager(aws)
    
    # Deploy models
    sagemaker_deploy = SageMakerDeployment(aws)
    
    # Setup real-time processing
    real_time_processor = RealTimeProcessor(aws)
    
    # Setup monitoring
    metrics = CloudMetrics(aws)
    
    print("🚀 AWS infrastructure setup complete!")