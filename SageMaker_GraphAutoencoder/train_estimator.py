# Initialize the estimatore
pytorch_estimator = PyTorch(entry_point='pytorch_training_file.py',
                            instance_type='ml.g4dn.2xlarge',
                            instance_count=1,
                            framework_version='1.4.0',
                            py_version='py3.6')
# Train the estimator
pytorch_estimator.fit()

# Deploy my estimator to a SageMaker Endpoint and get a Predictor
predictor = pytorch_estimator.deploy(instance_type='ml.m4.xlarge',
                                     initial_instance_count=1)

# Create predictions
response = predictor.predict(data)