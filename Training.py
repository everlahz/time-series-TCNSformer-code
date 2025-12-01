import os
import logging
import torch
import numpy as np
from collections import OrderedDict
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import random
from Models.loss import l2_reg_loss
from Models import utils, analysis
import warnings
from Models.hotmap import hotmap 
logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less


class BaseTrainer(object):

    def __init__(self, model, dataloader, device, loss_module, optimizer=None, l2_reg=None, print_interval=10,
                 console=True, print_conf_mat =False):

        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l2_reg = l2_reg  #L2 正则化系数
        self.print_interval = print_interval #打印日志的间隔步数
        self.printer = utils.Printer(console=console)
        self.print_conf_mat = print_conf_mat #是否打印混淆矩阵
        self.epoch_metrics = OrderedDict() #dict()的一个子类，保留了元素的插入顺序

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

    def print_callback(self, i_batch, metrics, prefix=''):

        total_batches = len(self.dataloader)

        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class SupervisedTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        """ 
            *args：收集所有按顺序传递的参数为一个元组，按照顺序传入，可以通过 args[index] 访问具体的参数。
            **kwargs：收集所有带有名字的参数为一个字典，以 key=value 形式传递的参数，可以通过 kwargs['key'] 访问具体的参数值。
        
        """
        super(SupervisedTrainer, self).__init__(*args, **kwargs)
        
        #判断config['loss_module']是否是 CrossEntropyLoss 类型。
        if isinstance(args[3], torch.nn.CrossEntropyLoss):
            # self.classification = True  # True if classification, False if regression
            self.analyzer = analysis.Analyzer(print_conf_mat=False)
        else:
            self.classification = False
        if kwargs['print_conf_mat']:
            self.analyzer = analysis.Analyzer(print_conf_mat=True)

    def train_epoch(self, epoch_num=None):
        
        #继承nn.Modul 启用训练模式，每次训练周期的开始调用 .train() 方法，以确保模型处于训练模式
        self.model = self.model.train()  #

        epoch_loss = 0  # 每一轮的总损失
        total_samples = 0  # 每一轮处理的样本总数
        
        #遍历 DataLoader 提供的每个数据批次
        for i, batch in enumerate(self.dataloader):

            X, targets, IDs = batch
            # M=torch.randn(X.shape).to(self.device)
            # #添加高斯噪声来进行数据增强
            # X = X + torch.randn(X.shape)*(0.0)
            
            #数据被传输到指定的设备（GPU）
            targets = targets.to(self.device)

            #导入数据，模型在这些数据上进行预测。
            predictions,attention_weights_all_layers= self.model(X.to(self.device), epoch_num) 
            hotmap(attention_weights_all_layers)
            
            #计算损失
            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(loss)  # mean loss (over samples) used for optimization
            
            #L2正则化
            if self.l2_reg:
                #将正则化损失添加到批次的平均损失中
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            
            #清除梯度
            self.optimizer.zero_grad()

            #反向传播
            total_loss.backward()

            # 梯度裁剪，max_norm 参数定义了允许的最大梯度范数值，如果梯度的范数超过 4.0，程序会相应地缩放梯度
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            
            #权重更新
            self.optimizer.step()

            #禁止梯度计算
            with torch.no_grad():
                #更新处理的总样本数和累计的损失
                total_samples += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        # if epoch_num % 20 == 0 and epoch_num > 1:
        #     att = self.model.get_att()
        #     sns.set()
        #     ax = sns.heatmap(att[0,0].detach().cpu().numpy(), fmt=".1f")
        #     plt.show()
        #     print(att)

        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        per_batch = {'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
        for i, batch in enumerate(self.dataloader):
            X, targets, IDs = batch
            targets = targets.to(self.device)
            predictions,attention_weights_all_layers = self.model(X.to(self.device), 0)
            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(loss)  # mean loss (over samples)

            per_batch['targets'].append(targets.cpu().numpy())
            predictions = predictions.detach()
            per_batch['predictions'].append(predictions.cpu().numpy())
            loss = loss.detach()
            per_batch['metrics'].append([loss.cpu().numpy()])
            per_batch['IDs'].append(IDs)

            metrics = {"loss": mean_loss}
            #if i % self.print_interval == 0:
                #ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                #self.print_callback(i, metrics, prefix='Evaluating ' + ending)

            total_samples += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
        probs = torch.nn.functional.softmax(predictions,dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        targets = np.concatenate(per_batch['targets'], axis=0).flatten()
        class_names = np.arange(probs.shape[1])  # TODO: temporary until I decide how to pass class names
        metrics_dict = self.analyzer.analyze_classification(predictions, targets, class_names)

        self.epoch_metrics['accuracy'] = metrics_dict['total_accuracy']  # same as average recall over all classes
        self.epoch_metrics['precision'] = metrics_dict['prec_avg']  # average precision over all classes

        '''
        if self.model.num_classes == 2:
            false_pos_rate, true_pos_rate, _ = sklearn.metrics.roc_curve(targets, probs[:, 1])  # 1D scores needed
            self.epoch_metrics['AUROC'] = sklearn.metrics.auc(false_pos_rate, true_pos_rate)

            prec, rec, _ = sklearn.metrics.precision_recall_curve(targets, probs[:, 1])
            self.epoch_metrics['AUPRC'] = sklearn.metrics.auc(rec, prec)
        '''
        return self.epoch_metrics, metrics_dict


def validate(val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""

    #logger.info("Evaluating on validation set ...")
    #eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, ConfMat = val_evaluator.evaluate(epoch, keep_all=True)
    #eval_runtime = time.time() - eval_start_time
    #logger.info("Validation runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(eval_runtime)))

    #global val_times
   # val_times["total_time"] += eval_runtime
    #val_times["count"] += 1
    #avg_val_time = val_times["total_time"] / val_times["count"]
    #avg_val_batch_time = avg_val_time / len(val_evaluator.dataloader)
    #avg_val_sample_time = avg_val_time / len(val_evaluator.dataloader.dataset)
    #logger.info("Avg val. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_val_time)))
    #logger.info("Avg batch val. time: {} seconds".format(avg_val_batch_time))
    #logger.info("Avg sample val. time: {} seconds".format(avg_val_sample_time))

    # print()
    # print_str = '\nVal: '
    # for k, v in aggr_metrics.items():
    #     tensorboard_writer.add_scalar('{}/val'.format(k), v, epoch)
    #     print_str += '{}: {:1f} | '.format(k, v)
    # logger.info(print_str)

    if config['key_metric'] in NEG_METRICS:
        condition = (aggr_metrics[config['key_metric']] < best_value)
    else:
        condition = (aggr_metrics[config['key_metric']] > best_value)
    if condition:
        best_value = aggr_metrics[config['key_metric']]
        utils.save_model(os.path.join(config['save_dir'], 'model_best.pth'), epoch, val_evaluator.model)
        best_metrics = aggr_metrics.copy()

        #pred_filepath = os.path.join(config['pred_dir'], 'best_predictions')
        # np.savez(pred_filepath, **per_batch)

    return aggr_metrics, best_metrics, best_value


def train_runner(config, model, trainer, val_evaluator,test_evaluator, path):
    # seed_value = 42
    # torch.manual_seed(seed_value)
    # torch.cuda.manual_seed(seed_value)
    # torch.cuda.manual_seed_all(seed_value)  # 如果你使用多 GPU
    # np.random.seed(seed_value)
    # random.seed(seed_value)
    epochs = config['epochs']
    optimizer = config['optimizer']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-3)
    loss_module = config['loss_module']
    start_epoch = 0
    total_start_time = time.time()
    tensorboard_writer = SummaryWriter('summary')
    best_value = 1e16
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}
    best_value_test = 1e16
    metrics_test = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics_test = {}
    save_best_model = utils.SaveBestModel()
    save_best_acc_model = utils.SaveBestACCModel()

    for epoch in tqdm(range(start_epoch, epochs), desc='Epoch',mininterval=0.01):
        training_time = time.time()
        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        # scheduler.step()
        logger.info("training time: %s" % (time.time() - training_time))
        print("training time: %s" % (time.time() - training_time))

        testing_time = time.time()
        aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config, best_metrics,
                                                              best_value, epoch)
        
        print("val time: %s" % (time.time() - testing_time))
        logger.info("val time: %s" % (time.time() - testing_time))
        aggr_metrics_test, best_metrics_test, best_value_test = validate(test_evaluator, tensorboard_writer, config, best_metrics_test,
                                                              best_value_test, epoch)
        #save_best_model(aggr_metrics_test['loss'], epoch, model, optimizer, loss_module, path)
        save_best_acc_model(aggr_metrics_test['accuracy'], epoch, model, optimizer, loss_module, path)

        metrics_names, metrics_values = zip(*aggr_metrics_val.items())
        metrics.append(list(metrics_values))

        print_str = 'Ep{}: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            if k != "epoch":
                tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
                print_str += '{}: {:4f} | '.format(k, v)
        for k, v in aggr_metrics_val.items():
            if k != "epoch":
                tensorboard_writer.add_scalar('{}/val'.format(k), v, epoch)
                print_str += 'val{}: {:4f} | '.format(k, v)
        for k, v in aggr_metrics_test.items():
            if k != "epoch":
                tensorboard_writer.add_scalar('{}/test'.format(k), v, epoch)
                print_str += 'test{}: {:4f} | '.format(k, v)
        logger.info(print_str)
    total_runtime = time.time() - total_start_time
    logger.info("Train Time: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))
    return