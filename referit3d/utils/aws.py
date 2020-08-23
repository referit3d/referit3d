import boto3

def establish_amt_client(region_name='us-east-1', aws_access_key=None, aws_secret_key=None):
    endpoint_url = 'https://mturk-requester.' + region_name + '.amazonaws.com'
    if aws_access_key is None or aws_secret_key is None: # Assume .config
        client = boto3.client('mturk', endpoint_url=endpoint_url, region_name=region_name)
    else:
        client = boto3.client('mturk', endpoint_url=endpoint_url, region_name=region_name,
                          aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
    return client


def print_utterances_of_worker(df, worker, utterance_column='utterance', worker_column='WorkerId'):
    u = df[df[worker_column] == worker][utterance_column]
    print(worker, ' # responses:', len(u))
    for i, j in zip(u.index, u):
        print(i, j)


def print_utterances_of_mask(df, mask, utterance_column='utterance', worker_column='WorkerId'):
    print('# responses:', mask.sum())

    if worker_column is not None:
        ndf = df[mask][[utterance_column, worker_column]]
        ndf = ndf.sort_values(worker_column)
    else:
        ndf = df[mask][utterance_column]

    for index, content in ndf.iterrows():
        if worker_column is not None:
            print(index, content[worker_column], content[utterance_column])
        else:
            print(index, content[utterance_column])
