

class Config:
    debug = False
    service_batch_size = 2
    clean_run = True

    tf_epoch_dir = 'data/model/epoch'
    tf_batch_dir = 'data/model/batch'
    tf_log_dir = 'data/model/log'

    data_embedding = 'data/embedding'

    def __init__(self):
        pass
