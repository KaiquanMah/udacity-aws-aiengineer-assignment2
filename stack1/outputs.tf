# module.aurora_serverless was defined in stack1/main.tf
# now we print relevant metadata for traceability when we need to validate OR modify certain AWS components/services
# diference between cluster_endpoint vs cluster_reader_endpoint???? write (and read) vs read permissions???? YES
output "db_endpoint" {
  value = module.aurora_serverless.cluster_endpoint
}

output "db_reader_endpoint" {
  value = module.aurora_serverless.cluster_reader_endpoint
}


# data.aws_vpc.default was defined in stack1/main.tf
output "vpc_id" {
  # value = module.vpc.vpc_security_group_ids
  # FIXED: Changed from module.vpc.vpc_id to the data source.
  value       = data.aws_vpc.default.id
}

# data.aws_subnets.default was defined in stack1/main.tf
output "default_subnet_ids" {
  description = "The IDs of all subnets in the default VPC."
  # FIXED: Changed from module.vpc.private_subnets to the data source.
  # Note: We are now getting ALL default subnets, not just private ones.
  value       = data.aws_subnets.default.ids
}

# REMOVED: The public_subnet_ids output is no longer relevant as we are using
# the data source which fetches all subnets without distinguishing public/private.
# output "private_subnet_ids" {
#   value = module.vpc.private_subnets
# }
# 
# output "public_subnet_ids" {
#   value = module.vpc.public_subnets
# }


# difference between db_endpoint vs aurora_endpoint - NONE; SAME VALUE
output "aurora_endpoint" {
  value = module.aurora_serverless.cluster_endpoint
}

# difference between endpoint vs ARN????
# ARN = AMZN resource name = resource ID, and grant permissions to tt resource to a user/role/svc (in IAM policy)
# endpoint (similar to 'IP') = for connection
output "aurora_arn" {
  value = module.aurora_serverless.database_arn
}

output "rds_secret_arn" {
  value = module.aurora_serverless.database_secretsmanager_secret_arn
}

output "s3_bucket_name" {
  # value = module.s3_bucket.s3_bucket_arn
  # FIXED: The module output for the bucket name is "s3_bucket_id".
  # The original value was pointing to the ARN, which is incorrect for the output name.
  value       = module.s3_bucket.s3_bucket_id
}