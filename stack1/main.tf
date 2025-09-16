provider "aws" {
  region = "us-east-1"  # Change this to your desired region # UPDATED
}


# ADDED TO FIX PERMISSIONS ISSUE
# This data block LOOKS UP the existing Default VPC. It does NOT create a new one.
# This is necessary to get the VPC ID and other details.
data "aws_vpc" "default" {
  default = true
}
# "vpc-0109f710cbe850abd" 


# This data block LOOKS UP the existing subnets within the Default VPC found above.
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# module "vpc" {
#   source  = "terraform-aws-modules/vpc/aws"
#   version = "~> 5.0"
# 
#   name = "bedrock-poc-vpc"
#   cidr = "10.0.0.0/16"
# 
#   azs             = ["us-east-1a", "us-east-1b", "us-east-1c"] # UPDATED
#   private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
#   public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
# 
#   enable_nat_gateway = true
#   single_nat_gateway = true
# 
#   enable_dns_hostnames = true
#   enable_dns_support   = true
# 
#   tags = {
#     Terraform   = "true"
#     Environment = "dev"
#   }
# }

module "aurora_serverless" {
  source = "../modules/database"

  cluster_identifier = "my-aurora-serverless"
  # vpc_id             = module.vpc.vpc_id 
  # subnet_ids         = module.vpc.private_subnets
  # UPDATED: Using the ID from the Default VPC we looked up.
  vpc_id = data.aws_vpc.default.id
  # UPDATED: Using the IDs from the subnets we looked up in the Default VPC.
  subnet_ids = data.aws_subnets.default.ids



  # Optionally override other defaults
  database_name    = "myapp"
  master_username  = "dbadmin"
  max_capacity     = 1
  min_capacity     = 0.5
  # allowed_cidr_blocks = ["10.0.0.0/16"]   
  # UPDATED: This now correctly allows traffic from within the Default VPC.
  allowed_cidr_blocks = [data.aws_vpc.default.cidr_block]
  
  # --- NEW: Tell the module to create this DB user ---
  # app_username = "app_user_01"

}

data "aws_caller_identity" "current" {}

locals {
  bucket_name = "bedrock-kb-${data.aws_caller_identity.current.account_id}"
}

module "s3_bucket" {
  source  = "terraform-aws-modules/s3-bucket/aws"
  version = "~> 3.0"

  bucket = local.bucket_name
  acl    = "private"
  force_destroy = true

  control_object_ownership = true
  object_ownership         = "BucketOwnerPreferred"

  versioning = {
    enabled = true
  }

  server_side_encryption_configuration = {
    rule = {
      apply_server_side_encryption_by_default = {
        sse_algorithm = "AES256"
      }
    }
  }

  # block_public_acls       = true
  # block_public_policy     = true
  # ignore_public_acls      = true
  # restrict_public_buckets = true
  # 2025.09.16 Fix to allow public access to VPC, to connect and run SQL on Aurora Serverless DB (required to complete assignment 2)
  #            Otherwise we are able to connect to Aurora DB on the 'Aurora and RDS' GUI, BUT unable to run queries (for some unexpected reason because permissions are not given for the assignment)
  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false

  tags = {
    Terraform   = "true"
    Environment = "dev"
  }
}
