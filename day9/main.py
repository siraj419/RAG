# Sync Programming vs Async Programming


# Statement 1 - 1s
# Statement 2 - 2s
# Statement 3 - 1s
#   = 4s

# Statement 1, Statement 2, Statement 3 
#  = 2s

# Prevents Code Blocking

# Async Queues

# Process API Server
#   - APis
    # broker - redis, rabbitmq, kafka
    # Amazon SQS, Google Pub/Sub, RQ
# Process - Worker - celery, rq

# Apis
#  - broker
# Worker

# Pub/Sub
#  - Publisher - Subscriber

# RAG

# Sync
# concurrency = no of tasks that can be done at the same time.


from app.tasks import answer


if __name__ == "__main__":
    user_question = input("Enter your question: ")
    print("Processing your question asynchronously...")
    print(answer.delay(user_question))
