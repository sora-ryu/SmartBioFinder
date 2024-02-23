# SmartBioFinder
This is final version of code repository that detects and tracks wildlife animals using thermal video cameras.

## How to run
### Download data
```
aws configure           # check .aws/config [region, output] and .aws/credentials [aws_access_key_id, aws_secret_access_key, aws_session_token]
aws s3 ls s3://nrel-globus-rsyu         # check s3 bucket
aws s3 sync s3://nrel-globus-rsyu/Thermal/2023-07-07_09_00_00_055/ ../videos/2023-07-07_09_00_00_055/       # Takes ~2mins to download one folder (15GB data)
```