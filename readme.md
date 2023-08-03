## Deploying to a Server

The MLflow instance proxies to S3 and RDS, so the only required credentials are the MLflow URI, Username and Password, stored in the .env file.

All MLflow calls are handled within `utils.py`, so further configuration is not required.

The Docker file contains the code, dataset and environment required to run the models...

### Instructions

- vast.ai lets you pull from Docker Hub. Use a READ ONLY access token instead of a real password!
- clone the repo (automated in vast.ai script)
- create the .env file in the root of the repo with MLFLOW_TRACKING_URI, \_USERNAME, \_PASSWORD, do this manually
- run the desired code
- remember to delete the instance when done (everything you need is on MLflow!)
