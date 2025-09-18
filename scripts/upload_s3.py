import os
import boto3
from botocore.exceptions import ClientError

def upload_files_to_s3(folder_path, bucket_name, prefix=""):
    # Initialize S3 client
    s3_client = boto3.client('s3')

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    # Walk through the directory
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            local_path = os.path.join(root, filename)
            
            # Calculate relative path for S3 key
            relative_path = os.path.relpath(local_path, folder_path)
            s3_key = os.path.join(prefix, relative_path).replace("\\", "/")

            try:
                # Upload the file to S3
                s3_client.upload_file(local_path, bucket_name, s3_key)
                print(f"Successfully uploaded {relative_path} to {bucket_name}/{s3_key}")
            except ClientError as e:
                print(f"Error uploading {relative_path}: {e}")

if __name__ == "__main__":
    # Folder path
    folder_path = "spec-sheets"
    
    # S3 bucket name
    bucket_name = "bedrock-kb-513101285646" # 2025.09.15 Mon AND Wed: created by terraform when it ran partially (until EC2 error came up) 
    # "bedrock-kb-awsengineerassignment2" # "bedrock-kb-975050171524"  # Replace with your actual bucket name
    
    # S3 prefix (optional)
    prefix = "spec-sheets" 
    
    upload_files_to_s3(folder_path, bucket_name, prefix)


# WHY IS IT THAT AFTER WE CREATED AURORA DB, BEDROCK KNOWLEDGE BASE (CONNECTED TO AURORA DB), UPLOADED DOCS TO S3, 
# WE NEED TO MANUALLY ADD FILES FROM S3 TO THE KNOWLEDGE BASE, THEN MANUALLY ADD FILES TO SYNC???? 
#    THE MANUAL ADDITION OF FILES DO NOT MAKE SENSE
#    Manual CUZ reading >> chunking docs >> from text generate embeddings >> ingest embeddings into Aurora DB
#           has a cost/needs resources
#           SO trigger/sync manually


