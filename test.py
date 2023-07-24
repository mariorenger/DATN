from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
# from recommenders.models.newsrec.models.resan import RESANModel
from recommenders.models.newsrec.models.naml import NAMLModel
# from recommenders.models.newsrec.models.renaml import RENAMLModel
from recommenders.models.newsrec.models.nrms import NRMSModel

from recommenders.models.newsrec.io.mind_all_iterator import MINDAllIterator
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set

import tensorflow as tf
import os

epochs = 10
seed = 42
batch_size = 32


# Options: demo, small, large
MIND_type = 'demo'

data_path = "./"

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
wordEmb_file = os.path.join(data_path, "utils", "embedding_all.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict_all.pkl")
vertDict_file = os.path.join(data_path, "utils", "vert_dict.pkl")
subvertDict_file = os.path.join(data_path, "utils", "subvert_dict.pkl")
yaml_file = os.path.join(data_path, "utils", r'nrms.yaml')

mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_type)

if not os.path.exists(train_news_file):
    download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)
    
if not os.path.exists(valid_news_file):
    download_deeprec_resources(mind_url, \
                               os.path.join(data_path, 'valid'), mind_dev_dataset)
if not os.path.exists(yaml_file):
    download_deeprec_resources(r'https://recodatasets.z20.web.core.windows.net/newsrec/', \
                               os.path.join(data_path, 'utils'), mind_utils)

hparams = prepare_hparams(yaml_file,
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file,
                          userDict_file=userDict_file,
                          vertDict_file=vertDict_file,
                          subvertDict_file=subvertDict_file,
                          batch_size=batch_size,
                          epochs=epochs)
print(hparams)


iterator = MINDAllIterator

model = NRMSModel(hparams, iterator, seed=seed)
# model = RENAMLModel(hparams, iterator, seed=seed)


model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)

# group_impr_indexes, group_labels, group_preds = model.run_slow_eval(valid_news_file, valid_behaviors_file)
# group_impr_indexes, group_labels, group_preds = model.run_fast_eval(valid_news_file, valid_behaviors_file)