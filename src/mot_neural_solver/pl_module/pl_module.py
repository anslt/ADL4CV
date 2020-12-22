import os
import os.path as osp

import pandas as pd

from torch_geometric.data import DataLoader

import torch
import numpy as np

from torch import optim as optim_module
from torch.optim import lr_scheduler as lr_sched_module
from torch.nn import functional as F

import pytorch_lightning as pl

from mot_neural_solver.data.mot_graph_dataset import MOTGraphDataset
from mot_neural_solver.models.mpn import MOTMPNet
from mot_neural_solver.models.resnet import resnet50_fc256, load_pretrained_weights
from mot_neural_solver.path_cfg import OUTPUT_PATH
from mot_neural_solver.utils.evaluation import compute_perform_metrics
from mot_neural_solver.tracker.mpn_tracker import MPNTracker

import matplotlib
import matplotlib.pyplot as plt

class MOTNeuralSolver(pl.LightningModule):
    """
    Pytorch Lightning wrapper around the MPN defined in model/mpn.py.
    (see https://pytorch-lightning.readthedocs.io/en/latest/lightning-module.html)

    It includes all data loading and train / val logic., and it is used for both training and testing models.
    """
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.validation_epoch = 0
        self.has_trained = True
        self.model, self.cnn_model = self.load_model()
    
    def forward(self, x):
        self.model(x)

    def load_model(self):
        cnn_arch = self.hparams['graph_model_params']['cnn_params']['arch']
        model =  MOTMPNet(self.hparams['graph_model_params']).cuda()

        cnn_model = resnet50_fc256(10, loss='xent', pretrained=True).cuda()
        load_pretrained_weights(cnn_model,
                                osp.join(OUTPUT_PATH, self.hparams['graph_model_params']['cnn_params']['model_weights_path'][cnn_arch]))
        cnn_model.return_embeddings = True

        return model, cnn_model

    def _get_data(self, mode, return_data_loader = True):
        assert mode in ('train', 'val', 'test')

        dataset = MOTGraphDataset(dataset_params=self.hparams['dataset_params'],
                                  mode=mode,
                                  cnn_model=self.cnn_model,
                                  splits= self.hparams['data_splits'][mode],
                                  logger=None)

        if return_data_loader and len(dataset) > 0:
            train_dataloader = DataLoader(dataset,
                                          batch_size = self.hparams['train_params']['batch_size'],
                                          shuffle = True if mode == 'train' else False,
                                          num_workers=self.hparams['train_params']['num_workers'])
            return train_dataloader
        
        elif return_data_loader and len(dataset) == 0:
            return []
        
        else:
            return dataset

    def train_dataloader(self):
        return self._get_data(mode = 'train')

    def val_dataloader(self):
        return self._get_data('val')

    def test_dataset(self, return_data_loader=False):
        return self._get_data('test', return_data_loader = return_data_loader)

    def configure_optimizers(self):
        optim_class = getattr(optim_module, self.hparams['train_params']['optimizer']['type'])
        optimizer = optim_class(self.model.parameters(), **self.hparams['train_params']['optimizer']['args'])

        if self.hparams['train_params']['lr_scheduler']['type'] is not None:
            lr_sched_class = getattr(lr_sched_module, self.hparams['train_params']['lr_scheduler']['type'])
            lr_scheduler = lr_sched_class(optimizer, **self.hparams['train_params']['lr_scheduler']['args'])

            return [optimizer], [lr_scheduler]

        else:
            return optimizer

    def _compute_loss(self, outputs, batch):
        att_regu = self.hparams['graph_model_params']['attention']['att_regu']
        # Define Balancing weight
        positive_vals = batch.edge_labels.sum()
        if positive_vals:
            pos_weight = (batch.edge_labels.shape[0] - positive_vals) / positive_vals
        else: # If there are no positives labels, avoid dividing by zero
            pos_weight = 0
        # Compute Weighted BCE:
        loss_class = 0
        num_steps_class = len(outputs['classified_edges'])
        for step in range(num_steps_class):
            loss_class += F.binary_cross_entropy_with_logits(outputs['classified_edges'][step].view(-1),
                                                            batch.edge_labels.view(-1),
                                                            pos_weight= pos_weight)
        
        att_loss_matrix = None
        att_regu_strength = self.hparams['graph_model_params']['attention']['att_regu_strength']
        att_loss = torch.FloatTensor([0]).to(loss_class.device)
        if att_regu:
            num_steps_attention = len(outputs['att_coefficients'])
            head_factor = self.hparams['graph_model_params']['attention']['attention_head_num']
            att_loss_matrix = torch.empty(size=(head_factor, num_steps_attention)).cuda()
            for step in range(num_steps_attention):
                for head in range(head_factor):
                    #weight = (batch.edge_labels.view(-1) == 0) + (batch.edge_labels.view(-1) == 1) * pos_weight
                    a = outputs['att_coefficients'][step][head].view(-1)
                    aa = torch.min(a,torch.ones_like(a).to(a.device))
                    att_loss_matrix[head, step] = F.binary_cross_entropy(
                        aa,
                        batch.edge_labels.view(-1))
                        #weight=weight)
            att_loss = torch.sum(att_loss_matrix) / head_factor
        else:
            att_loss = torch.FloatTensor([0]).to(loss_class.device)
            
        return {"loss": loss_class + att_regu_strength * att_loss , "loss_class": loss_class, "loss_regu" : att_loss}

    def training_step(self, batch, batch_idx):
        device = (next(self.model.parameters())).device
        batch.to(device)

        outputs = self.model(batch)
        loss_dic = self._compute_loss(outputs, batch)
        loss = loss_dic["loss"]
        logs = {**compute_perform_metrics(outputs, batch), **{'loss': loss}}
        log = {key + '/train': val for key, val in logs.items()}
        return {'loss': loss, 'log': log}

    def training_epoch_end(self, outputs):
        if self.has_trained == False:
            print("++++++++++++++")
            print("has trained is true.")
            self.has_trained = True
        return {}

    def validation_step(self, batch, batch_idx):
        device = (next(self.model.parameters())).device
        batch.to(device)
        outputs = self.model(batch)
        loss_dic = self._compute_loss(outputs, batch)
        loss = loss_dic["loss"]
        logs = {**compute_perform_metrics(outputs, batch), **{'loss': loss}}
        log = {key + '/val': val for key, val in logs.items()} 

        if self.has_trained == True:

            val_outputs = {"log": log}
            val_outputs["edge_attr"] = batch.edge_attr.detach()
            val_outputs["loss_regu"] = loss_dic["loss_regu"].detach()
            val_outputs["loss_class"] = loss_dic["loss_class"].detach()
            val_outputs["loss"] = loss_dic["loss"].detach()
            val_outputs["node_num"] = batch.x.size(0)

            attention = torch.stack(outputs['att_coefficients'], dim=0).detach().permute(2,1,0)
            val_outputs["attention"] = attention

            head_factor = self.hparams['graph_model_params']['attention']['attention_head_num']
            num_steps_attention = len(outputs['att_coefficients'])
            
            idx = batch.tracking_id.cpu()
            edge_id = batch.edge_index.detach().transpose(0,1).contiguous().cpu()
            attr = val_outputs["edge_attr"].cpu()
            labels = batch.edge_labels.detach().cpu()
            attention = attention.cpu()
            
            topk = self.hparams['visual']['topk']
            path = self.hparams['visual']['path']
            row = edge_id[:,0]
            col = edge_id[:,1]

            _, ind = torch.sort(row)

            edge_id = edge_id[ind]
            attention = attention[ind]
            attr = attr[ind]
            labels = labels[ind]
            
            #edge_id_cpu = edge_id.cpu()
            row = edge_id[:,0]
            col = edge_id[:,1]

            output, count = torch.unique_consecutive(row,return_counts=True)
            cum_count = torch.cumsum(count, dim=0)

            attention_step = attention.mean(dim=1)

            step_size = attention_step.size(1)
            val_res_step = torch.zeros(step_size,topk).to(attention.device)

            for k in range(step_size):
                
                #cal_num = 0 
                val_res = torch.zeros(topk)
                
                for i in range(output.size(0)):

                    if i == 0:
                        attack_i = attention_step[range(cum_count[i]),k]
                        temp = 0
                    else:
                        attack_i = attention_step[range(cum_count[i-1],cum_count[i]),k]
                        temp = cum_count[i-1]
                    val_i, ind_i = torch.sort(attack_i, descending=True)

                    if ind_i.shape[0] >= topk:
                        #cal_num += torch.sum(labels[ind_i+temp])
                        val_topk = val_i[range(topk)]
                        val_res += torch.cumsum(val_topk, dim=0) 
                
                val_res_step[k,:] = val_res

            attention_mean_last = [attention.mean(dim=[1,2]), attention[:,:,-1].mean(-1)]

            val_res_mean_last = torch.zeros(2,topk).to(attention.device)
            exists_topk_mean_last = torch.zeros(2,topk).to(attention.device)
            tracking_exists_topk_mean_last = torch.zeros(2,topk).to(attention.device)
            tracking_percentage_topk_mean_last = torch.zeros(2,topk).to(attention.device)

            for k, attention0 in enumerate(attention_mean_last):
                
                                               ### set
                val_res = torch.zeros(topk)
                exists_topk = torch.zeros(topk)
                percentage_topk = torch.zeros(topk)
                tracking_exists_topk = torch.zeros(topk)
                tracking_percentage_topk = torch.zeros(topk)
                cal_num = 0 
                num = 0

                for i in range(output.size(0)):

                    if i == 0:
                        attack_i = attention0[range(cum_count[i])]
                        temp = 0
                    else:
                        attack_i = attention0[range(cum_count[i-1],cum_count[i])]
                        temp = cum_count[i-1]
                    val_i, ind_i = torch.sort(attack_i, descending=True)

                    if ind_i.shape[0] >= topk:
                        cal_num += torch.sum(labels[ind_i+temp])
                        num += torch.any(row[ind_i+temp] > col[ind_i+temp]).type(torch.FloatTensor) + torch.any(row[ind_i+temp] < col[ind_i+temp]).type(torch.FloatTensor)
                        ind_topk = ind_i[range(topk)] + temp
                        val_topk = val_i[range(topk)]
                        val_res += torch.cumsum(val_topk, dim=0)

                        labels_i_topk = torch.cumsum(labels[ind_topk], dim=0)
                        exists_topk += labels_i_topk

                        tracking_i = idx[output[i].type(torch.LongTensor)] == idx[col[ind_topk].type(torch.LongTensor)]
                        tracking_i_cumsum = torch.cumsum(tracking_i, dim=0)
                        tracking_percentage_topk += tracking_i_cumsum
                        tracking_exists_topk += tracking_i_cumsum > 0
                        
                        val_res_mean_last[k] = val_res
                        exists_topk_mean_last[k] = exists_topk
                        tracking_exists_topk_mean_last[k] = tracking_exists_topk
                        tracking_percentage_topk_mean_last[k] = tracking_percentage_topk

            
            
            val_outputs["cal_edge_num"] = cal_num
            val_outputs["edge_num"] = num
            val_outputs["val_res_step"] = val_res_step.to(device)
            val_outputs["val_res"] = val_res_mean_last.to(device)
            val_outputs["exists_topk"] = exists_topk_mean_last.to(device)
            val_outputs["tracking_exists_topk"] = tracking_exists_topk_mean_last.to(device)
            val_outputs["tracking_percentage_topk"] = tracking_percentage_topk_mean_last.to(device)
            
        return val_outputs

    def validation_epoch_end(self, val_outputs):
        """
        att_statistics = val_outputs[-1]["att_statistics"]
        quantity = ['loss','mean','variance','min','max','min mean', 'min var', 'max mean', 'max var','topk accuracy','trajectory','emb_diff','time']
        t = 0
        for i in quantity:
            print(i,':',att_statistics[t])
            t += 1
        """
        if self.has_trained == True:

            self.validation_epoch += 1
            edge_attr = []
            attention = []
            cal_edge_num = 0
            edge_num = 0
            node_num = 0

            val_res_step = torch.zeros_like(val_outputs[0]["val_res_step"])
            val_res = torch.zeros_like(val_outputs[0]["val_res"])
            exists_topk = torch.zeros_like(val_outputs[0]["exists_topk"])
            tracking_percentage_topk = torch.zeros_like(val_outputs[0]["tracking_percentage_topk"])
            tracking_exists_topk = torch.zeros_like(val_outputs[0]["tracking_exists_topk"])
            loss = []
            loss_class = []
            loss_regu = []
            
            for val_output in val_outputs:
                edge_attr += [val_output["edge_attr"]]
                attention += [val_output["attention"]]
                loss += [val_output["loss"]]
                loss_class += [val_output["loss_class"]]
                loss_regu += [val_output["loss_regu"]]
                edge_num += val_output["edge_num"]
                cal_edge_num = val_output["cal_edge_num"] + cal_edge_num
                node_num += val_output["node_num"]
                    
                val_res_step += val_output["val_res_step"]
                val_res += val_output["val_res"]
                exists_topk += val_output["exists_topk"]
                tracking_exists_topk += val_output["tracking_exists_topk"]
                tracking_percentage_topk += val_output["tracking_percentage_topk"]
            
            edge_attr = torch.cat(edge_attr, dim = 0)
            attention = torch.cat(attention, dim = 0)

            val_res_step /= edge_num
            val_res /= edge_num
            exists_topk /= cal_edge_num
            tracking_exists_topk /= cal_edge_num
            tracking_percentage_topk /= cal_edge_num * torch.cumsum(torch.FloatTensor(np.arange(self.hparams['visual']['topk'])+1), dim=0).to(attention.device).unsqueeze(0)

            print("\nEpoch:"+str(self.validation_epoch))
            print(loss_class)
            print(torch.stack(loss_class, dim=0).mean())
            print(loss_regu)
            print(torch.stack(loss_regu, dim=0).mean())
            print(val_res_step)
            print(val_res)
            print(exists_topk)
            print(tracking_exists_topk)
            print(tracking_percentage_topk)
            
            path = self.hparams['visual']['path']
            np.save(path+"val_res_step_"+str(self.validation_epoch)+".npy",val_res_step.detach().cpu().numpy())
            np.save(path+"val_res_"+str(self.validation_epoch)+".npy",val_res.detach().cpu().numpy())
            np.save(path+"exists_topk_"+str(self.validation_epoch)+".npy",exists_topk.detach().cpu().numpy())
            np.save(path+"tracking_exists_topk_"+str(self.validation_epoch)+".npy",tracking_exists_topk.detach().cpu().numpy())
            np.save(path+"tracking_percentage_topk_"+str(self.validation_epoch)+".npy",tracking_percentage_topk.detach().cpu().numpy())

            self._plot(edge_attr.cpu(), attention.cpu(), self.validation_epoch, path, 'last')
            self._plot(edge_attr.cpu(), attention.cpu(), self.validation_epoch, path, 'mean')

        metrics = pd.DataFrame([val["log"] for val in val_outputs]).mean(axis=0).to_dict()
        metrics = {metric_name: torch.as_tensor(metric) for metric_name, metric in metrics.items()}
        return {'val_loss': metrics['loss/val'], 'log': metrics}

    def _plot(self,attr,attention,epoch,path=None,mode="last"):
        assert attr.size(1)==6
        assert mode in ["mean","last"]
        
        if mode == "mean":
            attention = attention.mean(dim=-1)
        else:
            attention = attention[:,:,-1]
        index = [1,2,3,4,0,5]
        attr=attr[:,index]
        x_labels=["distance","x_distance","y_distance","log_height_ratio","log_width_ratio","time_difference","embedding_distance"]
        y_label="attention"
        k = attention.size(1) ### k need to be 2
        
        for i in range(7):
            if i == 0:
                x = torch.sqrt(attr[:,0] * attr[:,0] + attr[:,1] * attr[:,1])
            else:
                x = attr[:,i-1]
                
            ymax = torch.max(attention)
            ylim_max = ymax + 0.05 if ymax < 0.95 else 1
            
            fig = plt.figure(figsize=(40,25))
            ax = fig.add_subplot(121)

            plt.plot(x, attention[:,0], 'ro', markersize=2)

            ax.set_xlabel(x_labels[i], fontsize=30)
            ax.set_ylabel(y_label, fontsize=30)
            ax.set_title("mode:" + mode + " ,epoch:" + str(epoch) + " ,node 1. " + x_labels[i] + " vs " + y_label, fontsize=30)
            ax.set_ylim(0,ylim_max)
            
            ax = fig.add_subplot(122)

            plt.plot(x, attention[:,1], 'bo', markersize=2)

            ax.set_xlabel(x_labels[i], fontsize=30)
            ax.set_ylabel(y_label, fontsize=30)
            ax.set_title("mode:" + mode + " ,epoch:" + str(epoch) + " ,node 1. " + x_labels[i] + " vs " + y_label, fontsize=30)
            ax.set_ylim(0,ylim_max)
            
            file = "epoch_" + str(epoch) + "_" + mode + "_" + x_labels[i] + ".png"
            if path is not None:
                file = path+file
            plt.savefig(file)
            #plt.show()
            plt.close()

    def track_all_seqs(self, output_files_dir, dataset, use_gt = False, verbose = False):
        tracker = MPNTracker(dataset=dataset,
                             graph_model=self.model,
                             use_gt=use_gt,
                             eval_params=self.hparams['eval_params'],
                             dataset_params=self.hparams['dataset_params'])

        constraint_sr = pd.Series(dtype=float)
        for seq_name in dataset.seq_names:
            print("Tracking", seq_name)
            if verbose:
                print("Tracking sequence ", seq_name)

            os.makedirs(output_files_dir, exist_ok=True)
            _, constraint_sr[seq_name] = tracker.track(seq_name, output_path=osp.join(output_files_dir, seq_name + '.txt'))

            if verbose:
                print("Done! \n")


        constraint_sr['OVERALL'] = constraint_sr.mean()

        return constraint_sr
