from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
from mamba_nas.performance import model_performance
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach()
                loss = criterion(pred, label.long().squeeze())
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()

        accuracy = cal_accuracy(predictions, trues)
        macro_f1 = f1_score(
            trues,
            predictions,
            average="macro",
            zero_division=0,
        )

        self.model.train()
        return total_loss, accuracy, macro_f1

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy, val_macro_f1 = self.vali(
                vali_data, vali_loader, criterion
            )
            test_loss, test_accuracy, test_macro_f1 = self.vali(
                test_data, test_loader, criterion
            )

            print(
                "Epoch: {0}, Steps: {1} | "
                "Train Loss: {2:.3f} | "
                "Vali Loss: {3:.3f} Vali Acc: {4:.4f} "
                "Vali Macro-F1: {5:.4f} | "
                "Test Loss: {6:.3f} Test Acc: {7:.4f} "
                "Test Macro-F1: {8:.4f}"
                .format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    vali_loss,
                    val_accuracy,
                    val_macro_f1,
                    test_loss,
                    test_accuracy,
                    test_macro_f1,
                )
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)
        macro_f1 = f1_score(trues, predictions, average='macro', zero_division=0)
        performance = model_performance(self.model, test_loader, self.device)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        print('macro_f1:{}'.format(macro_f1))
        print('parameters:{}'.format(performance['parameters']))
        print('macs:{}'.format(performance['macs']))
        print('flops:{}'.format(performance.get('flops')))
        print('profile_macs:{}'.format(performance.get('profile_macs')))
        print('profile_flops:{}'.format(performance.get('profile_flops')))
        print('latency_median_ms:{}'.format(performance.get('latency_median_ms')))
        print('latency_p90_ms:{}'.format(performance.get('latency_p90_ms')))
        print('throughput_samples_per_sec:{}'.format(performance.get('throughput_samples_per_sec')))
        print('peak_memory_bytes:{}'.format(performance.get('peak_memory_bytes')))
        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('macro_f1:{}'.format(macro_f1))
        f.write('\n')
        f.write('parameters:{}'.format(performance['parameters']))
        f.write('\n')
        f.write('macs:{}'.format(performance['macs']))
        f.write('\n')
        f.write('macs_definition:{}'.format(performance['macs_definition']))
        f.write('\nflops:{}'.format(performance.get('flops')))
        f.write('\nprofile_macs:{}'.format(performance.get('profile_macs')))
        f.write('\nprofile_flops:{}'.format(performance.get('profile_flops')))
        f.write('\nlatency_median_ms:{}'.format(performance.get('latency_median_ms')))
        f.write('\nlatency_p90_ms:{}'.format(performance.get('latency_p90_ms')))
        f.write('\nthroughput_samples_per_sec:{}'.format(performance.get('throughput_samples_per_sec')))
        f.write('\npeak_memory_bytes:{}'.format(performance.get('peak_memory_bytes')))
        f.write('\nefficiency_definition:{}'.format(performance.get('efficiency_definition')))
        f.write('\n')
        f.write('\n')
        f.close()
        return
