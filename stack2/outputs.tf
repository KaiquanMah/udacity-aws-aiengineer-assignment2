# READ from bedrock_kb (CHILD MODULE)'s outputs.tf
output "bedrock_knowledge_base_id" {
  value = module.bedrock_kb.id
}

output "bedrock_knowledge_base_arn" {
  value = module.bedrock_kb.arn
}
