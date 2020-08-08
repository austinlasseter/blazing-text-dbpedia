# this function triggers an endpoint to a model trained on SageMaker with BlazingText

import boto3
import json
import re

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

# a simple tokenizer function
def review_to_words(review):
    words = REPLACE_NO_SPACE.sub("", review.lower())
    words = REPLACE_WITH_SPACE.sub(" ", words)
    return words

# primary function
def lambda_handler(event, context):
    # lambda receives the input from the web app as an event in json format
    sentences = event['body']
    tokenized_sentences = review_to_words(sentences)
    payload = {"instances" : [tokenized_sentences]}

    # The SageMaker runtime is what allows us to invoke the endpoint that we've created.
    runtime = boto3.Session().client('sagemaker-runtime')

    # Now we use the SageMaker runtime to invoke our endpoint, sending the text we were given
    response = runtime.invoke_endpoint(EndpointName = 'blazingtext-2020-08-08-09-19-31-404',# The name of the endpoint we created
                                       ContentType = 'application/json',                 # The data format that is expected by BlazingText
                                       Body = json.dumps(payload))

    # The response is an HTTP response whose body contains the result of our inference
    output = json.loads(response['Body'].read().decode('utf-8'))
    prob = output[0]['prob'][0]*100
    label = output[0]['label'][0].split('__label__')[1]
    output = 'The predicted label is {} with a probability of {:.1f}%'.format(label, prob)

    # we return the output in a format expected by API Gateway
    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'body' : output
    }
