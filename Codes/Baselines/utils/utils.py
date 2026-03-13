from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
from datetime import datetime as dt 
from tensorboardX import SummaryWriter
import os

import json
import pandas as pd

class Recorder():

    def __init__(self, early_step):
        self.max = {'metric': 0}
        self.cur = {'metric': 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1
        print("current", self.cur)
        return self.judge()

    def judge(self):
        # 依据metrix指标，即macro f1，判断是否需要继续训练
        if self.cur['metric'] > self.max['metric']:
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print("Max", self.max)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def metrics(y_true, y_pred):
    all_metrics = {}

    # AUCs can fail in some degenerate cases; keep them guarded
    try:
        all_metrics['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    except:
        all_metrics['auc'] = 0

    try:
        all_metrics['spauc'] = roc_auc_score(y_true, y_pred, average='macro', max_fpr=0.1)
    except:
        all_metrics['spauc'] = 0
    
    y_pred = np.array(y_pred)

    print("Raw prediction min:", y_pred.min())
    print("Raw prediction max:", y_pred.max())
    print("Raw prediction mean:", y_pred.mean())

    y_pred = (y_pred > 0.40).astype(int)

    all_metrics['metric'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    f1_vals = f1_score(y_true, y_pred, average=None, zero_division=0)
    rec_vals = recall_score(y_true, y_pred, average=None, zero_division=0)
    pre_vals = precision_score(y_true, y_pred, average=None, zero_division=0)

    all_metrics['f1_real'] = f1_vals[0]
    all_metrics['f1_fake'] = f1_vals[1] if len(f1_vals) > 1 else 0

    all_metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    all_metrics['recall_real'] = rec_vals[0]
    all_metrics['recall_fake'] = rec_vals[1] if len(rec_vals) > 1 else 0

    all_metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    all_metrics['precision_real'] = pre_vals[0]
    all_metrics['precision_fake'] = pre_vals[1] if len(pre_vals) > 1 else 0

    all_metrics['acc'] = accuracy_score(y_true, y_pred)
    
    return all_metrics

def data2gpu(batch, use_cuda, data_type):
    if use_cuda:
        if data_type == 'news_env':
            batch_data = {
                'content': batch[0].cuda(),
                'content_masks': batch[1].cuda(),
                'entity': batch[2].cuda(),
                'entity_masks': batch[3].cuda(),
                'label': batch[4].cuda(),
                'year': batch[5].cuda(),
                'emotion': batch[6].cuda()
                }
        elif data_type == 'online' or data_type == 'roll_online':
            batch_data = {
                'content': batch[0].cuda(),
                'content_masks': batch[1].cuda(),
                'image': batch[2].cuda(),
                'raw_text': batch[3],          # TEXT MUST stay on CPU (list of strings)
                'label': batch[4].cuda(),
                'weight': batch[5].cuda(),
                'id': batch[6].cuda(),
                'year': batch[7].cuda(),
                'year_season': batch[8].cuda(),
            }

        else:
            print('error data type!')
            exit()           
    else:
        if data_type == 'news_env':
            batch_data = {
                'content': batch[0],
                'content_masks': batch[1],
                'entity': batch[2],
                'entity_masks': batch[3],
                'label': batch[4],
                'year': batch[5],
                'emotion': batch[6]
                }
        elif data_type == 'online' or data_type == 'roll_online':
            batch_data = {
                'content': batch[0],
                'content_masks': batch[1],
                'image': batch[2],
                'raw_text': batch[3],
                'label': batch[4],
                'weight': batch[5],
                'id': batch[6],
                'year': batch[7],
                'year_season': batch[8],
            }

        else:
            print('error data type!')
            exit()         
    return batch_data

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def get_split_path(split_level, data_type, root_path, time, data_name):
    """ 输入分割级别、数据类型、当前时间等信息，返回目标文件路径
    """
    if split_level == 'monthly':
        if data_type == 'news_env':
            file_path = root_path + data_name
            return file_path
        elif data_type == 'online':
            file_path = os.path.join(root_path, 'month_'+str(time), data_name)
            return file_path
        elif data_type == 'roll_online':
            file_path = os.path.join(root_path, 'roll_month_'+str(time), data_name)
            return file_path        
        else:
            print('No match data type!')
            exit()
    elif split_level == 'seasonal':
        if data_type == 'news_env':
            file_path = root_path + data_name
            return file_path
        elif data_type == 'online':
            file_path = os.path.join(root_path, 'season_'+str(time), data_name)
            return file_path
        elif data_type == 'roll_online':
            file_path = os.path.join(root_path, 'roll_season_'+str(time), data_name)
            return file_path        
        else:
            print('No match data type!')
            exit()
    else:
        print('Error split level!')
        exit(1) 

def get_tensorboard_writer(config):
    """
    Build a writable log dir and return a SummaryWriter.
    Prefer $EANN_LOG_ROOT or fall back to ~/biswajit/seasonal_logs to avoid PermissionError.
    """
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(dt.now())
    # Choose a base log root that is guaranteed writable
    base_log_root = os.environ.get(
        "EANN_LOG_ROOT",
        os.path.join(os.path.expanduser("~"), "biswajit", "seasonal_logs")
    )

    # Use config['tensorboard_dir'] if it is absolute and writable; otherwise use base_log_root
    tb_root = config.get('tensorboard_dir', './seasonal_logs')
    if not os.path.isabs(tb_root):
        tb_root = base_log_root

    #writer_dir = os.path.join(config['tensorboard_dir'], config['model_name'] + '_' + config['data_name'], TIMESTAMP)
    writer_dir = os.path.join(
        tb_root,
        config['model_name'] + '_' + config['data_name'],
        TIMESTAMP
    )

    # Ensure directory exists before constructing SummaryWriter
    os.makedirs(writer_dir, exist_ok=True)
    writer = SummaryWriter(logdir=writer_dir, flush_secs=5)
   
    return writer

def process_test_results(test_file_path, test_res_path, label, pred, id, ae, acc):
    """ 处理并写入test结果
    """
    test_result = []
    test_df = pd.read_json(test_file_path)
    for index in range(len(label)):
        cur_res = {}
        cur_id = id[index]
        m = test_df['id'] == int(cur_id)
        if not m.any():
            # If id not found, skip this entry (or log if preferred)
            continue
        cur_data = test_df.loc[m].iloc[0]
        #cur_data = test_df[test_df['id'] == int(cur_id)].iloc[0]
        for (key, val) in cur_data.items(): 
            cur_res[key] = val
        cur_res['pred'] = pred[index]
        cur_res['ae'] = ae[index]
        cur_res['acc'] = acc[index]

        test_result.append(cur_res)

    json_str = json.dumps(test_result, indent=4, ensure_ascii=False, cls=NpEncoder)

    with open(test_res_path, 'w', encoding='utf-8') as f:
        f.write(json_str)
