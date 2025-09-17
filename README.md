# udacity-aws-aiengineer-assignment2

Code for assignment 2
- Deploy stack1 terraform which deploys an aurora serverless DB
- For some unexpected reason, my account was able to use Query Editor to connect to Aurora Servless DB, but not given permissions to use Query Editor to run queries (which was neeed by the actual assignment)
- As a workaround: Check aurora serverless DB, find and update config (including allowing inbound/outbound Postgres connection) to connect to it from the Terminal (still unable to connect from DataGrip IDE even with the right drivers and somehow with the right settings)
- Prepare aurora DB with the SQL commands, required for knowledge base to work with later
- Deploy stack2 terraform which deploys a Bedrock Knowledge Base
- Upload 5 assignment PDFs to S3
- Sync knowledge base with the PDFs in S3 (had this unexpected issue where I could only add 1 document at a time, instead of adding the whole S3 bucket, or the subfolder containing all 5 PDFs)
- Then update bedrock_utils.py based on the assignment
  - For the RAG workflow in particular, there were 'huge changes'
  - Answer_with_sources
    - The full function that runs this whole RAG pipeline is in the ‘answer_with_sources’ function.
    - It classifies and validates a user query using the ‘valid_prompt’ function.
    - Then ‘query_knowledge_base’ and ‘_normalise_result’ of the retrieved chunks.
    - Use the chunks to ‘_build_prompt’ which contains the user query with context and sources.
    - Then call the Bedrock model and parse the model output using ‘_parse_answer’, to display in the Terminal.

Challenges faced
- Missing steps in the assignment, which I had to figure out myself and asking on forums to continue debugging
- Permission issues - e.g. with the Aurora DB connection, Query Editor
- Overall, I had to top up more steps and debug over multiple days to create the final solution. I was not able to use the original repo (https://github.com/udacity/cd13926-Building-Generative-AI-Applications-with-Amazon-Bedrock-and-Python-project-solution/tree/main) as it is.
