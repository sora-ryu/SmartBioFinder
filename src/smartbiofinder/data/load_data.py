import boto3
import glob
import argparse

"""
Class including all related functions for accessing AWS bucket.

How to run:
python load_data.py 
"""

class DataBucket():
    def __init__(self, args):
        self.bucket = args.bucket
        self.key = args.key
        self.file = args.file
        self.s3 = boto3.client('s3')
    
    # Read and list objects from the bucket
    def list_objects(self):
        response = self.s3.list_objects(Bucket=self.bucket, MaxKeys=10)
        return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load Data with S3')

    # Credentials for the AWS Access - Automatically made by reading ~/.aws/credentials file
    # A Path for upload/download objects
    parser.add_argument('--bucket', default='nrel-globus-rsyu', type=str, help='A bucket to access')
    parser.add_argument('--key', default='LiDAR/CSV/capture20230706_104157/', type=str, help='A path for the object in AWS bucket')
    parser.add_argument('--file', default='LiDAR/CSV/capture20230706_104157/', type=str, help='A local path for the object')

    # Action to perform
    parser.add_argument('--list_obj', default='xxx', type=str, help='Add argument to list the objects')

    args = parser.parse_args()
    db = DataBucket(args)

    if args.list_obj is not None:
        response = db.list_objects()
        print(response)