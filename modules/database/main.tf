resource "aws_rds_cluster" "aurora_serverless" {
  cluster_identifier      = var.cluster_identifier
  engine                  = "aurora-postgresql"
  engine_mode             = "provisioned"
  engine_version          = var.engine_version
  database_name           = var.database_name
  master_username         = var.master_username
  master_password         = random_password.master_password.result
  enable_http_endpoint    = true
  skip_final_snapshot     = true
  apply_immediately       = true

  allow_major_version_upgrade = true

  serverlessv2_scaling_configuration {
    max_capacity             = var.max_capacity
    min_capacity             = var.min_capacity
  }

  vpc_security_group_ids = [aws_security_group.aurora_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.aurora.name
}

resource "aws_rds_cluster_instance" "aurora_instance" {
  cluster_identifier = aws_rds_cluster.aurora_serverless.id
  # round1
  # instance_class     = "db.t3.medium"
  # │ Error: creating RDS Cluster (my-aurora-serverless) Instance (tf-20250915022434192200000001): operation error RDS: CreateDBInstance, https response error StatusCode: 400, RequestID: ab8b4c04-50bb-42a3-8179-a0fda6bf1d21, api error InvalidParameterCombination: The instance class that you specified doesn't support the HTTP endpoint for using RDS Data API.
  # │ 
  # │   with module.aurora_serverless.aws_rds_cluster_instance.aurora_instance,
  # │   on ../modules/database/main.tf line 24, in resource "aws_rds_cluster_instance" "aurora_instance":
  # │   24: resource "aws_rds_cluster_instance" "aurora_instance" {
  # round2
  # module.aurora_serverless.aws_rds_cluster_instance.aurora_instance: Creating...
  # module.aurora_serverless.aws_rds_cluster_instance.aurora_instance: Still creating... [10s elapsed]
  # module.aurora_serverless.aws_rds_cluster_instance.aurora_instance: Still creating... [3m0s elapsed]
  # module.aurora_serverless.aws_rds_cluster_instance.aurora_instance: Still creating... [3m10s elapsed]
  # module.aurora_serverless.aws_rds_cluster_instance.aurora_instance: Still creating... [3m20s elapsed]
  # instance_class     = "db.serverless" # "db.t3.medium"
  # round3 - no instance_class - for aws aurora serverless
  # need instance_class
  # round4
  instance_class     = "db.serverless"
  engine             = aws_rds_cluster.aurora_serverless.engine
  engine_version     = aws_rds_cluster.aurora_serverless.engine_version

  
  # 2025.09.16
  # --- NEW ---
  publicly_accessible = true
}

resource "aws_db_subnet_group" "aurora" {
  name       = "${var.cluster_identifier}-subnet-group"
  subnet_ids = var.subnet_ids

  tags = {
    Name = "${var.cluster_identifier}-subnet-group"
  }
}

resource "aws_security_group" "aurora_sg" {
  name        = "${var.cluster_identifier}-sg"
  description = "Security group for Aurora Serverless"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.cluster_identifier}-sg"
  }
}

# New resources for secret management
resource "random_password" "master_password" {
  length  = 16
  special = true
}

resource "aws_secretsmanager_secret" "aurora_secret" {
  name = "${var.cluster_identifier}"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "aurora_secret_version" {
  secret_id = aws_secretsmanager_secret.aurora_secret.id
  secret_string = jsonencode({
    dbClusterIdentifier = aws_rds_cluster.aurora_serverless.cluster_identifier
    password            = random_password.master_password.result
    engine              = aws_rds_cluster.aurora_serverless.engine
    port                = 5432
    host                = aws_rds_cluster.aurora_serverless.endpoint
    username            = aws_rds_cluster.aurora_serverless.master_username
    db                  = aws_rds_cluster.aurora_serverless.database_name
  })
}







# # 2025.09.16
# # --- NEW: variables for app user creation ---
# variable "app_username" {
#   description = "Optional: Application DB username to create (Aurora PostgreSQL). If null, skip creating."
#   type        = string
#   default     = null
# }
# # --- NEW: random password for the app user (keep special=false to avoid SQL quoting issues) ---
# resource "random_password" "app_user_password" {
#   count   = var.app_username == null ? 0 : 1
#   length  = 20
#   special = false
# }
# 
# # --- NEW: Secret for the app user. Use the 'rds-db-credentials/' prefix so the standard Query Editor IAM policy can read it ---
# resource "aws_secretsmanager_secret" "app_user_secret" {
#   count                   = var.app_username == null ? 0 : 1
#   name                    = "rds-db-credentials/${var.cluster_identifier}-${var.app_username}"
#   recovery_window_in_days = 0
# }
# 
# resource "aws_secretsmanager_secret_version" "app_user_secret_version" {
#   count        = var.app_username == null ? 0 : 1
#   secret_id    = aws_secretsmanager_secret.app_user_secret[0].id
#   secret_string = jsonencode({
#     username = var.app_username
#     password = random_password.app_user_password[0].result
#   })
# }
# 
# # --- NEW: Create the DB user via RDS Data API and grant privileges ---
# resource "null_resource" "create_app_user" {
#   count = var.app_username == null ? 0 : 1
# 
#   # Ensure DB is ready before we run SQL
#   depends_on = [
#     aws_rds_cluster_instance.aurora_instance
#   ]
# 
#   # Re-run if any of these change
#   triggers = {
#     cluster_arn  = aws_rds_cluster.aurora_serverless.arn
#     secret_arn   = aws_secretsmanager_secret.aurora_secret.arn  # master creds secret you already create above
#     db_name      = aws_rds_cluster.aurora_serverless.database_name
#     app_username = var.app_username
#     pw_checksum  = sha256(random_password.app_user_password[0].result)
#   }
# 
#   provisioner "local-exec" {
#     # Requires AWS CLI v2 in the environment where you run `terraform apply`
#     interpreter = ["/bin/bash", "-lc"]
#     command = <<-EOT
#       set -euo pipefail
# 
#       CLUSTER_ARN="${aws_rds_cluster.aurora_serverless.arn}"
#       MASTER_SECRET_ARN="${aws_secretsmanager_secret.aurora_secret.arn}"
#       DB="${aws_rds_cluster.aurora_serverless.database_name}"
#       APP_USER="${var.app_username}"
#       APP_PW="${random_password.app_user_password[0].result}"
# 
#       # 1) Create ROLE if not exists (single statement with a DO block so it's idempotent)
#       aws rds-data execute-statement \
#         --resource-arn "$CLUSTER_ARN" \
#         --secret-arn "$MASTER_SECRET_ARN" \
#         --database "$DB" \
#         --sql "DO $$ BEGIN
#                  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '$APP_USER') THEN
#                    CREATE ROLE \"$APP_USER\" LOGIN PASSWORD '$APP_PW';
#                  END IF;
#                END $$;"
# 
#       # 2) Basic privileges on the database
#       aws rds-data execute-statement \
#         --resource-arn "$CLUSTER_ARN" \
#         --secret-arn "$MASTER_SECRET_ARN" \
#         --database "$DB" \
#         --sql "GRANT CONNECT, TEMP ON DATABASE $DB TO \"$APP_USER\";"
# 
#       # 3) Schema usage + DML on all current objects in public schema
#       aws rds-data execute-statement \
#         --resource-arn "$CLUSTER_ARN" \
#         --secret-arn "$MASTER_SECRET_ARN" \
#         --database "$DB" \
#         --sql "GRANT USAGE ON SCHEMA public TO \"$APP_USER\";"
# 
#       aws rds-data execute-statement \
#         --resource-arn "$CLUSTER_ARN" \
#         --secret-arn "$MASTER_SECRET_ARN" \
#         --database "$DB" \
#         --sql "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO \"$APP_USER\";"
# 
#       aws rds-data execute-statement \
#         --resource-arn "$CLUSTER_ARN" \
#         --secret-arn "$MASTER_SECRET_ARN" \
#         --database "$DB" \
#         --sql "GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public TO \"$APP_USER\";"
# 
#       # 4) Default privileges for future objects
#       aws rds-data execute-statement \
#         --resource-arn "$CLUSTER_ARN" \
#         --secret-arn "$MASTER_SECRET_ARN" \
#         --database "$DB" \
#         --sql "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO \"$APP_USER\";"
# 
#       aws rds-data execute-statement \
#         --resource-arn "$CLUSTER_ARN" \
#         --secret-arn "$MASTER_SECRET_ARN" \
#         --database "$DB" \
#         --sql "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO \"$APP_USER\";"
#     EOT
#   }
# }
# 
# # --- NEW: handy outputs (optional) ---
# output "app_user_secret_arn" {
#   value       = try(aws_secretsmanager_secret.app_user_secret[0].arn, null)
#   description = "Secrets Manager secret ARN that stores the app user's username/password."
# }
