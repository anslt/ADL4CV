import os
import os.path as osp

import pandas as pd

from torch_geometric.data import DataLoader

import torch

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

class MOTNeuralSolver(pl.LightningModule):
    """
    Pytorch Lightning wrapper around the MPN defined in model/mpn.py.
    (see https://pytorch-lightning.readthedocs.io/en/latest/lightning-module.html)

    It includes all data loading and train / val logic., and it is used for both training and testing models.
    """
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
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
    
        if att_regu:
            loss_att = 0
            num_steps_attention = len(outputs['att_coefficients'])
            head_factor = self.hparams['graph_model_params']['attention']['attention_head_num']
            att_regu_strength = self.hparams['graph_model_params']['attention']['att_regu_strength']
            for step in range(num_steps_attention):
                for head in range(head_factor):
                    loss_att += F.binary_cross_entropy_with_logits(outputs['att_coefficients'][step][head].view(-1),
                                                                              batch.edge_labels.view(-1),
                                                                              pos_weight= pos_weight)
            loss_att = loss_att/head_factor
            return loss_class + att_regu_strength*loss_att
        return loss_class

    def training_step(self, batch, batch_idx):
        device = (next(self.model.parameters())).device
        batch.to(device)

        #print(batch.graph_size)
        #print(batch.x.size())
        #print(batch.frame.size())
        #print(batch.tracking_id.size())
        #print(batch.frame)
        #print(batch.tracking_id)

        outputs = self.model(batch)
        loss = self._compute_loss(outputs, batch)
        logs = {**compute_perform_metrics(outputs, batch), **{'loss': loss}}
        log = {key + '/train': val for key, val in logs.items()}
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        device = (next(self.model.parameters())).device
        batch.to(device)
        outputs = self.model(batch)
        loss = self._compute_loss(outputs, batch)
        logs = {**compute_perform_metrics(outputs, batch), **{'loss': loss}}
        log = {key + '/val': val for key, val in logs.items()} 
        val_outputs = log
        
        head_factor = self.hparams['graph_model_params']['attention']['attention_head_num']
        num_steps_attention = len(outputs['att_coefficients'])
        
        
        
        att_statistics = torch.empty(size=(13,head_factor,num_steps_attention)).cuda()
        positive_vals = batch.edge_labels.sum()
        if positive_vals:
            pos_weight = (batch.edge_labels.shape[0] - positive_vals) / positive_vals
        else: # If there are no positives labels, avoid dividing by zero
            pos_weight = 0
        k = 5
        x, idx, edge_attr,label,time,identity = batch.x, batch.edge_index, batch.edge_attr, batch.edge_labels, batch.frame, batch.tracking_id
        time = time.float()
        centers = torch.unique(idx)
        centers = centers[0:100]
        center_num = len(centers)
        for step in range(num_steps_attention):
            for head in range(head_factor):
                a = outputs['att_coefficients'][step][head].view(-1)
                ### attention loss matrix ### 
                att_statistics[0,head,step] = F.binary_cross_entropy_with_logits(outputs['att_coefficients'][step][head].view(-1),
                                                                              batch.edge_labels,
                                                                              pos_weight= pos_weight) 
                ### attention loss matrix ### 
                ### attention mean matrix ###
                att_statistics[1,head,step] = torch.mean(outputs['att_coefficients'][step][head].view(-1))
                ### attention mean matrix ###
                ### attention variance matrix ###
                att_statistics[2,head,step] = torch.var(outputs['att_coefficients'][step][head].view(-1))  
                ### attention variance matrix ###
                ### attention minimum matrix ###
                att_statistics[3,head,step] = torch.min(outputs['att_coefficients'][step][head].view(-1))        
                ### attention minimum matrix ###
                ### attention maximum matrix ###
                att_statistics[4,head,step] = torch.max(outputs['att_coefficients'][step][head].view(-1))        
                ### attention maximum matrix ###
                accuracy = 0
                dis_rate = 0
                time_rate = 0
                same = 0
                mins = torch.empty(size=(center_num,))
                maxs = torch.empty(size=(center_num,))
                count = 0
                for center in centers:
                    mask = (center == idx[0])
                    b = a.clone()
                    b[~mask] = -1 
                    _, high_topk_mask = torch.topk(b,k)                         # boolean tensor [M,] M: number of edges
                    high_topk_neighbors = idx[1][high_topk_mask]                # Long tensor [k,] 
                    
                    c = a.clone()
                    c[~mask] = 2
                    _, low_topk_mask = torch.topk(c,k,largest= False)
                    low_topk_neighbors = idx[1][low_topk_mask]
                    ### min mean, min var, max mean, max var###
                    mins[count] = torch.min(a[mask])
                    maxs[count] = torch.max(a[mask])
                    count += 1
                    ### min mean, min var, max mean, max var###
                    ### topk accuracy ###
                    accuracy += torch.sum(label[high_topk_mask])/2
                    ### topk accuarcy ###                
                    ### trajectory ###
                    same += torch.sum(identity[center] == identity[high_topk_neighbors])/k
                    ### trajectory ###
                    ### node embedding ###                    
                    dis1 = torch.sum(torch.norm(x[center]-x[high_topk_neighbors],dim=1))
                    dis2 = torch.norm(x[center]-x[torch.arange(x.size(0))!=center],dim=1,p=None) 
                    value,_ = torch.topk(dis2,k,largest= False)
                    dis2 = torch.sum(value)
                    dis_rate += dis1/dis2
                    ### node embedding ###
                    ### time ###
                    time_diff_high = torch.mean(torch.abs(time[center]-time[high_topk_neighbors]))
                    time_diff_low = torch.mean(torch.abs(time[center]-time[low_topk_neighbors]))
                    time_rate += time_diff_high/time_diff_low
                    ### time ###
                    
                att_statistics[5,head,step] = torch.mean(mins)
                att_statistics[6,head,step] = torch.var(mins)
                att_statistics[7,head,step] = torch.mean(maxs)
                att_statistics[8,head,step] = torch.var(maxs)             
                accuracy /= center_num
                att_statistics[9,head,step] = accuracy
                same /= center_num
                att_statistics[10,head,step] = same
                dis_rate /= center_num
                att_statistics[11,head,step] = dis_rate
                time_rate /= center_num
                att_statistics[12,head,step] = time_rate
                
        val_outputs["att_statistics"] = att_statistics
        return val_outputs

    def validation_epoch_end(self, val_outputs):
        att_statistics = val_outputs[-1]["att_statistics"]
        quantity = ['loss','mean','variance','min','max','min mean', 'min var', 'max mean', 'max var','topk accuracy','trajectory','emb_diff','time']
        t = 0
        for i in quantity:
            print(i,':',att_statistics[t])
            t += 1
        
        metrics = pd.DataFrame(val_outputs).mean(axis=0).to_dict()
        metrics = {metric_name: torch.as_tensor(metric) for metric_name, metric in metrics.items()}
        return {'val_loss': metrics['loss/val'], 'log': metrics}

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
