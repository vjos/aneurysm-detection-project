## Deploying to a Server

The MLflow instance proxies to S3 and RDS, so the only required credentials are the MLflow URI, Username and Password, stored in the .env file.

All MLflow calls are handled within `utils.py`, so further configuration is not required.

The Docker file contains the code, dataset and environment required to run the models. Simply pull or build the image to a different repo, and run the code from there.

vast.ai lets you pull from Docker Hub. Use a READ ONLY access token instead of a real password!
