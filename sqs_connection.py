import boto.sqs
import boto3
import json

###################### Credentiale ###########################
region_name = 'us-east-2'
queue_name = 'test-debug'
max_queue_messages = 0
message_bodies = []
aws_access_key_id = '.................'
aws_secret_access_key = '.......................'

##################### Connection stablished with SQS ##########
sqs = boto3.resource('sqs', region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key)
        
queue = sqs.get_queue_by_name(QueueName=queue_name)

################### Get Message From SQS #######################
messages = queue.receive_messages(MaxNumberOfMessages=10, WaitTimeSeconds=1)
#open("test_debug.txt", "w").close()
file = open('test_debug.txt', 'w')
file.close()
success_msg = ''
while len(messages) > 0:
    for message in messages:
        ### Write/append  message into file ###
        with open('test_debug.txt', 'a') as the_file:
            item = message.body
            the_file.write(f"{item}\n")
            the_file.close()
            ### Delete a message from sqs #####
            #message.delete()
            success_msg = 'Successfully extract data from sqs(test_debug)'

    messages = queue.receive_messages(MaxNumberOfMessages=10, WaitTimeSeconds=1)

