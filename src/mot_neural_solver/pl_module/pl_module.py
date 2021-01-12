import os
import os.path as osp
import numpy as np

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
        weight = (batch.edge_labels.view(-1) == 0) + (batch.edge_labels.view(-1) == 1) * pos_weight
        for step in range(num_steps_class):
            loss_class += F.binary_cross_entropy(outputs['classified_edges'][step].view(-1),
                                                        batch.edge_labels.view(-1),
                                                        weight= weight)    
        if att_regu:
            loss_att = 0
            num_steps_attention = len(outputs['att_coefficients'])
            head_factor = self.hparams['graph_model_params']['attention']['attention_head_num']
            att_regu_strength = self.hparams['graph_model_params']['attention']['att_regu_strength']
            for step in range(num_steps_attention):
                for head in range(head_factor):
                    a = outputs['att_coefficients'][step][head].view(-1)
                    aa = torch.min(2*a,torch.full(a.shape,1).cuda())
                    loss_att += F.binary_cross_entropy(aa,batch.edge_labels.view(-1))
            loss_att = loss_att/head_factor
            return loss_class + att_regu_strength*loss_att
        return loss_class

    def training_step(self, batch, batch_idx):
        device = (next(self.model.parameters())).device
        batch.to(device)

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
        
        accumulated_fn = torch.zeros(len(outputs['mask'])+2,).cuda()
        for i in range(len(outputs['mask'])):
            mask = outputs['mask'][i]
            accumulated_fn[i]=torch.sum(batch.edge_labels.view(-1)[~mask])
        accumulated_fn[-1] = torch.sum(batch.edge_labels.view(-1))
        accumulated_fn[-2] = len(batch.edge_labels)
        val_outputs['dynamic'] = accumulated_fn
        val_outputs["mask"] = torch.sum(1-outputs['mask'],dim=0)
        return val_outputs

    def validation_epoch_end(self, val_outputs):
        dynamic = torch.zeros_like(val_outputs[0]['dynamic'])
        mask = torch.zeros_like(val_outputs[0]['mask'])

        for k in range(len(val_outputs)):
            dynamic += val_outputs[k]['dynamic']
            mask += val_outputs[k]['mask']

        print(dynamic)
        print(mask)

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
