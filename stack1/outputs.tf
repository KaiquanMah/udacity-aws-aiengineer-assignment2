output "db_endpoint" {
  value = module.aurora_serverless.cluster_endpoint
}

output "db_reader_endpoint" {
  value = module.aurora_serverless.cluster_reader_endpoint
}

output "vpc_id" {
  # value = module.vpc.vpc_security_group_ids
  # FIXED: Changed from module.vpc.vpc_id to the data source.
  value       = data.aws_vpc.default.id
}


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

output "aurora_endpoint" {
  value = module.aurora_serverless.cluster_endpoint
}

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