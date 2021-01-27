import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_mean, scatter_max, scatter_add
import torch_scatter

from mot_neural_solver.models.mlp import MLP


class MetaLayer(torch.nn.Module):
    """
    Core Message Passing Network Class. Extracted from torch_geometric, with minor modifications.
    (https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html)
    """
    def __init__(self, edge_model=None, node_model=None,use_attention=True):
        """
        Args:
            edge_model: Callable Edge Update Model
            node_model: Callable Node Update Model
            use_attention: attention is used or not
        """
        super(MetaLayer, self).__init__()

        self.edge_model = edge_model
        self.node_model = node_model
        self.use_attention = use_attention
        self.reset_parameters()
        
    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """
        Does a single node and edge feature vectors update.
        Args:
            x: node features matrix
            edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
            graph adjacency (i.e. edges)
            edge_attr: edge features matrix (ordered by edge_index)

        Returns: Updated Node and Edge Feature matrices

        """
        row, col = edge_index

        # Edge Update
        if self.edge_model is not None:
            edge_attr = self.edge_model(x[row], x[col], edge_attr)

        # Node Update
        if self.node_model is not None:
            if self.use_attention:
                x,a = self.node_model(x, edge_index, edge_attr)
                return x,edge_attr,a
            else: 
                x = self.node_model(x, edge_index, edge_attr)
                return x, edge_attr

    def __repr__(self):
        return '{}(edge_model={}, node_model={})'.format(self.__class__.__name__, self.edge_model, self.node_model)

class EdgeModel(nn.Module):
    """
    Class used to peform the edge update during Neural message passing
    """
    def __init__(self, edge_mlp):
        super(EdgeModel, self).__init__()
        self.edge_mlp = edge_mlp

    def forward(self, source, target, edge_attr):
        out = torch.cat([source, target, edge_attr], dim=1)
        return self.edge_mlp(out)

class AttentionModel(nn.Module):
    def __init__(self, configs):
        super(AttentionModel, self).__init__()

        self.alpha = configs["alpha"]
        self.in_features = configs["in_features"]
        self.attention_head_num = configs["attention_head_num"]

        # add cuda directly
        self.aa = nn.Parameter(torch.empty(size=(self.attention_head_num,2*self.in_features))).cuda() #[k,64]
        nn.init.xavier_uniform_(self.aa.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x,row,col):
        xx = torch.cat([x[row], x[col]], dim=1)                                                          # cat([M,d=32],[M,d=32])->[M,d=64]
        e = self.leakyrelu(torch.matmul(self.aa,xx.T))                                                   # [k,64]*[64,M]->[k,M]
        a = torch_scatter.composite.scatter_softmax(e,row, dim=1, eps=1e-12)                             # [k,M]->[k,M]
        return a

class TimeAwareNodeModel(nn.Module):
    """
    Class used to peform the node update during Neural mwssage passing
    """

    def __init__(self, flow_out_mlp, flow_in_mlp, node_mlp, node_agg_fn, time_aware=True, attention_configs=None):
        super(TimeAwareNodeModel, self).__init__()        
        self.flow_out_mlp = flow_out_mlp
        self.flow_in_mlp = flow_in_mlp
        self.node_mlp = node_mlp
        self.node_agg_fn = node_agg_fn
        self.time_aware = time_aware
        
        if attention_configs is not None and attention_configs["use_attention"] == True:
            self.use_attention = attention_configs["use_attention"]
            self.head_factor = attention_configs["attention_head_num"]
            self.attention_out = AttentionModel(attention_configs)
            if self.time_aware:
                self.attention_in = AttentionModel(attention_configs)
        else:
            self.use_attention = False
            self.head_factor = 1

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index                                                                               # [M,]
        if self.time_aware:
            flow_out_mask = row < col                                                                       # [M1,]
            flow_out_row, flow_out_col = row[flow_out_mask], col[flow_out_mask]                             # [M,] -> [M1,]
            flow_out_input = torch.cat([x[flow_out_row], edge_attr[flow_out_mask]], dim=1)                  # [M1,d=32+2*16]
            #flow_out = torch.empty(size=(self.head_factor,flow_out_input.shape[0],32)) 
            flow_out = []

            flow_in_mask = row > col                                                                        # [M2,]
            flow_in_row, flow_in_col = row[flow_in_mask], col[flow_in_mask]                                 # [M,] -> [M2,]
            flow_in_input = torch.cat([x[flow_in_row], edge_attr[flow_in_mask]], dim=1)                     # [M2,32+2*16]
            #flow_in = torch.empty(size=(self.head_factor,flow_in_input.shape[0],32))
            flow_in = []
            for i in range(self.head_factor):
                #flow_out[i] = self.flow_out_mlp[i](flow_out_input)  # [M,48]->[k,M,32]
                #flow_in[i] = self.flow_in_mlp[i](flow_in_input)
                flow_out.append(self.flow_out_mlp[i](flow_out_input).cuda())
                flow_in.append(self.flow_in_mlp[i](flow_in_input).cuda())                                     
            flow_out = torch.stack(flow_out,dim=0)                                                          # [k,M1,d=32] 
            flow_in = torch.stack(flow_in,dim=0)                                                            # [k,M2,d=32]
 
            if self.use_attention:
                a_out = self.attention_out(x,flow_out_row,flow_out_col)                                     # [k,M1]
                a_in = self.attention_in(x,flow_in_row,flow_out_col)                                        # [k,M2]
                flow_out = torch.mul(flow_out.transpose(0,2),a_out.transpose(0,1)).transpose(0,1).transpose(1,2).reshape(flow_out_row.shape[0],-1)
                # [d=32,M1,k] [M1,k] -> [d=32,M1,k] -> [M1,32,k] -> [M1,k,32] -> [M1,32k]
                flow_in = torch.mul(flow_in.transpose(0,2),a_in.transpose(0,1)).transpose(0,1).transpose(1,2).reshape(flow_in_row.shape[0],-1) 
                # [d=32,M2,k] [M2,k] -> [d=32,M2,k] -> [M2,32,k] -> [M2,k,32] -> [M2,32k]
                flow_out =  self.node_agg_fn(flow_out, flow_out_row, x.size(0))                             # [M1,32k] -> [N,32k]
                flow_in =  self.node_agg_fn(flow_in, flow_in_row, x.size(0))  # [M,32k] -> [N,32k]          # [M2,32k] -> [N,32k]
                
                a = torch.empty(size=(self.head_factor,row.shape[0])).cuda()                                # [M,] 
                a.T[flow_out_mask]=a_out.T                                                                  # put [M1,] to [M,]
                a.T[flow_in_mask]=a_in.T                                                                    # put [M2,] to [M,]
            else:
                flow_out = self.node_agg_fn(torch.squeeze(flow_out,0), flow_out_row, x.size(0))             # [1,M1,d=32] -> [M1,d=32] -> [N,32] 
                flow_in = self.node_agg_fn(torch.squeeze(flow_in,0), flow_in_row, x.size(0))                # [1,M2,d=32] -> [M2,d=32] -> [N,32]
            flow = torch.cat([flow_in, flow_out], dim=1)                                                    # if attention: cat([N,32k],[N,32k]) -> [N,64k]
                                                                                                            # else: cat([N,32],[N32]) -> [N,64]
        else:
            flow_input = torch.cat([x[row], edge_attr], dim=1)                                              # cat([M,d=32],[M,d=16*2]) -> [M,64]
            flow = []       
            for i in range(self.head_factor):
                flow.append(self.flow_out_mlp[i](flow_input))                                               # [M,d=64]->[k,M,d=32]
            flow = torch.stack(flow,dim=0)                                                                  # [k,M,d=32]
            
            if self.use_attention:
                a = self.attention_out(x,row,col)                                                           # [k,M]
                flow = torch.mul(flow.transpose(0,2),a.transpose(0,1)).transpose(0,1).transpose(1,2).reshape(row.shape[0],-1)
                # [d=32,M,k] [M,k] -> [d=32,M,k] -> [M,32,k] -> [M,k,32] -> [M,32k]
                flow =  self.node_agg_fn(flow, row, x.size(0))                                              # [M,32k] -> [N,32k]
            else:
                flow = self.node_agg_fn(torch.squeeze(flow,0), row, x.size(0))                              # [1,M,32] -> [M,32] -> [N,32]
        if self.use_attention:
            return self.node_mlp(flow),a                                                                    # [N,32] [k,M]
        else: return self.node_mlp(flow)                                                                    # [N,32]


class MLPGraphIndependent(nn.Module):
    """
    Class used to to encode (resp. classify) features before (resp. after) neural message passing.
    It consists of two MLPs, one for nodes and one for edges, and they are applied independently to node and edge
    features, respectively.

    This class is based on: https://github.com/deepmind/graph_nets tensorflow implementation.
    """

    def __init__(self, edge_in_dim = None, node_in_dim = None, edge_out_dim = None, node_out_dim = None,
                 node_fc_dims = None, edge_fc_dims = None, dropout_p = None, use_batchnorm = None):
        super(MLPGraphIndependent, self).__init__()

        if node_in_dim is not None :
            self.node_mlp = MLP(input_dim=node_in_dim, fc_dims=list(node_fc_dims) + [node_out_dim],
                                dropout_p=dropout_p, use_batchnorm=use_batchnorm)
        else:
            self.node_mlp = None

        if edge_in_dim is not None :
            self.edge_mlp = MLP(input_dim=edge_in_dim, fc_dims=list(edge_fc_dims) + [edge_out_dim],
                                dropout_p=dropout_p, use_batchnorm=use_batchnorm)
        else:
            self.edge_mlp = None

    def forward(self, edge_feats = None, nodes_feats = None):

        if self.node_mlp is not None:
            out_node_feats = self.node_mlp(nodes_feats)

        else:
            out_node_feats = nodes_feats

        if self.edge_mlp is not None:
            out_edge_feats = self.edge_mlp(edge_feats)

        else:
            out_edge_feats = edge_feats

        return out_edge_feats, out_node_feats

class MOTMPNet(nn.Module):
    """
    Main Model Class. Contains all the components of the model. It consists of of several networks:
    - 2 encoder MLPs (1 for nodes, 1 for edges) that provide the initial node and edge embeddings, respectively,
    - 4 update MLPs (3 for nodes, 1 per edges used in the 'core' Message Passing Network
    - 1 edge classifier MLP that performs binary classification over the Message Passing Network's output.

    This class was initially based on: https://github.com/deepmind/graph_nets tensorflow implementation.
    """

    def __init__(self, model_params, bb_encoder = None):
        """
        Defines all components of the model
        Args:
            bb_encoder: (might be 'None') CNN used to encode bounding box apperance information.
            model_params: dictionary contaning all model hyperparameters
        """
        super(MOTMPNet, self).__init__()

        self.node_cnn = bb_encoder
        self.model_params = model_params
        self.time_aware = model_params["time_aware"]
        self.network_split = model_params["network_split"]
        self.use_attention = model_params["attention"]["use_attention"]
        if self.use_attention:
            self.use_att_regu = model_params["attention"]["att_regu"]
            self.attention_head_num = model_params["attention"]["attention_head_num"]

        # Define Encoder and Classifier Networks
        encoder_feats_dict = model_params['encoder_feats_dict']
        classifier_feats_dict = model_params['classifier_feats_dict']

        self.encoder = MLPGraphIndependent(**encoder_feats_dict)
        self.classifier = MLPGraphIndependent(**classifier_feats_dict)

        self.num_enc_steps = model_params['num_enc_steps']

        # Define the 'Core' message passing network (i.e. node and edge update models)
        self.MPNet = self._build_core_MPNet(model_params=model_params, encoder_feats_dict=encoder_feats_dict)

        self.num_class_steps = model_params['num_class_steps']
        self.num_attention_steps = model_params['num_attention_steps']
        self.graph_pruning = model_params["dynamical_graph"]['graph_pruning']
        if self.graph_pruning:
            self.first_prune_step = model_params["dynamical_graph"]['first_prune_step']
            self.prune_factor = model_params["dynamical_graph"]['prune_factor']
            self.prune_frequency = model_params["dynamical_graph"]['prune_frequency']
            self.prune_mode = model_params["dynamical_graph"]['mode']
            self.prune_min_edge =  model_params["dynamical_graph"]['prune_min_edge']
            self.edge_mlp_output_dim = model_params['edge_model_feats_dict']['fc_dims'][-1]
            assert self.prune_mode in ["classifier node wise", "classifier naive", "similarity node wise", "similarity naive"]
        
        
    def _build_core_MPNet(self, model_params, encoder_feats_dict):
        """
        Builds the core part of the Message Passing Network: Node Update and Edge Update models.
        Args:
            model_params: dictionary contaning all model hyperparameters
            encoder_feats_dict: dictionary containing the hyperparameters for the initial node/edge encoder
        """

        # Define an aggregation operator for nodes to 'gather' messages from incident edges
        if self.use_attention:
            node_agg_fn = 'sum' 
        else: node_agg_fn = model_params['node_agg_fn']
        assert node_agg_fn.lower() in ('mean', 'max', 'sum'), "node_agg_fn can only be 'max', 'mean' or 'sum'."
        
        if node_agg_fn == 'mean':
            node_agg_fn = lambda out, row, x_size: scatter_mean(out, row, dim=0, dim_size=x_size)

        elif node_agg_fn == 'max':
            node_agg_fn = lambda out, row, x_size: scatter_max(out, row, dim=0, dim_size=x_size)[0]

        elif node_agg_fn == 'sum':
            node_agg_fn = lambda out, row, x_size: scatter_add(out, row, dim=0, dim_size=x_size)

        # Define all MLPs involved in the graph network
        # For both nodes and edges, the initial encoded features (i.e. output of self.encoder) can either be
        # reattached or not after each Message Passing Step. This affects MLPs input dimensions
        self.reattach_initial_nodes = model_params['reattach_initial_nodes']
        self.reattach_initial_edges = model_params['reattach_initial_edges']


        edge_factor = 2 if self.reattach_initial_edges else 1
        node_factor = 2 if self.reattach_initial_nodes else 1
        time_factor = 2 if self.time_aware else 1


        edge_model_in_dim = node_factor * 2 * encoder_feats_dict['node_out_dim'] + edge_factor * encoder_feats_dict[
            'edge_out_dim']
        node_model_in_dim = node_factor * encoder_feats_dict['node_out_dim'] + encoder_feats_dict['edge_out_dim']

        # Define all MLPs used within the MPN
        edge_model_feats_dict = model_params['edge_model_feats_dict']
        node_model_feats_dict_time = model_params['node_model_feats_dict_time']
        node_model_feats_dict = model_params['node_model_feats_dict'] 
        
        # Define attention property
        attention_configs = model_params["attention"]
        attention_configs["in_features"] = node_factor * encoder_feats_dict['node_out_dim']
        head_factor = attention_configs["attention_head_num"] if self.use_attention else 1

        edge_mlp = MLP(input_dim=edge_model_in_dim,
                       fc_dims=edge_model_feats_dict['fc_dims'],
                       dropout_p=edge_model_feats_dict['dropout_p'],
                       use_batchnorm=edge_model_feats_dict['use_batchnorm']).cuda()

        node_mlp = MLP(input_dim=encoder_feats_dict['node_out_dim'] * head_factor * time_factor,
                       fc_dims=node_model_feats_dict['fc_dims'],
                       dropout_p=node_model_feats_dict['dropout_p'],
                       use_batchnorm=node_model_feats_dict['use_batchnorm']).cuda()

        if self.network_split:
            network = []
            for step in range(1, self.num_enc_steps + 1):
                flow_out_mlp = []
                for i in range(head_factor):
                    flow_out_mlp.append(MLP(input_dim=node_model_in_dim,
                                            fc_dims=node_model_feats_dict_time['fc_dims'],
                                            dropout_p=node_model_feats_dict_time['dropout_p'],
                                            use_batchnorm=node_model_feats_dict_time['use_batchnorm']).cuda())
                if self.time_aware:
                    flow_in_mlp = []
                    for i in range(head_factor):
                        flow_in_mlp.append(MLP(input_dim=node_model_in_dim,
                                               fc_dims=node_model_feats_dict_time['fc_dims'],
                                               dropout_p=node_model_feats_dict_time['dropout_p'],
                                               use_batchnorm=node_model_feats_dict_time['use_batchnorm']).cuda())
                    network += [MetaLayer(edge_model=EdgeModel(edge_mlp=edge_mlp),
                                     node_model=TimeAwareNodeModel(flow_out_mlp=flow_out_mlp,
                                                                   flow_in_mlp=flow_in_mlp,
                                                                   node_mlp=node_mlp,
                                                                   node_agg_fn=node_agg_fn,
                                                                   time_aware=self.time_aware,
                                                                   attention_configs=attention_configs),
                                     use_attention=self.use_attention)]
                else:
                    network += [MetaLayer(edge_model=EdgeModel(edge_mlp=edge_mlp),
                                     node_model=TimeAwareNodeModel(flow_out_mlp=flow_out_mlp,
                                                                   flow_in_mlp=None,
                                                                   node_mlp=node_mlp,
                                                                   node_agg_fn=node_agg_fn,
                                                                   time_aware=self.time_aware,
                                                                   attention_configs=attention_configs),
                                     use_attention=self.use_attention)]

            return network
        else:

            flow_out_mlp = []
            for i in range(head_factor):
                flow_out_mlp.append(MLP(input_dim=node_model_in_dim,
                                        fc_dims=node_model_feats_dict_time['fc_dims'],
                                        dropout_p=node_model_feats_dict_time['dropout_p'],
                                        use_batchnorm=node_model_feats_dict_time['use_batchnorm']).cuda())
            if self.time_aware:
                flow_in_mlp = []
                for i in range(head_factor):
                    flow_in_mlp.append(MLP(input_dim=node_model_in_dim,
                                       fc_dims=node_model_feats_dict_time['fc_dims'],
                                       dropout_p=node_model_feats_dict_time['dropout_p'],
                                       use_batchnorm=node_model_feats_dict_time['use_batchnorm']).cuda())
                return MetaLayer(edge_model=EdgeModel(edge_mlp=edge_mlp),
                                 node_model=TimeAwareNodeModel(flow_out_mlp=flow_out_mlp,
                                                               flow_in_mlp=flow_in_mlp,
                                                               node_mlp=node_mlp,
                                                               node_agg_fn=node_agg_fn,
                                                               time_aware=self.time_aware,
                                                               attention_configs=attention_configs),
                                  use_attention=self.use_attention)
            else:
                return MetaLayer(edge_model=EdgeModel(edge_mlp = edge_mlp),
                                 node_model=TimeAwareNodeModel(flow_out_mlp = flow_out_mlp,
                                                               flow_in_mlp = None,
                                                               node_mlp = node_mlp,
                                                               node_agg_fn = node_agg_fn,
                                                               time_aware=self.time_aware,
                                                               attention_configs=attention_configs),
                                  use_attention=self.use_attention)

    def forward(self, data):
        """
        Provides a fractional solution to the data association problem.
        First, node and edge features are independently encoded by the encoder network. Then, they are iteratively
        'combined' for a fixed number of steps via the Message Passing Network (self.MPNet). Finally, they are
        classified independently by the classifiernetwork.
        Args:
            data: object containing attribues
              - x: node features matrix
              - edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
                graph adjacency (i.e. edges) (i.e. sparse adjacency)
              - edge_attr: edge features matrix (sorted by edge apperance in edge_index)

        Returns:
            classified_edges: list of unnormalized node probabilites after each MP step
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x_is_img = len(x.shape) == 4
        if self.node_cnn is not None and x_is_img:
            x = self.node_cnn(x)

            emb_dists = nn.functional.pairwise_distance(x[edge_index[0]], x[edge_index[1]]).view(-1, 1)
            edge_attr = torch.cat((edge_attr, emb_dists), dim = 1)

        # Encoding features step
        latent_edge_feats, latent_node_feats = self.encoder(edge_attr, x)
        initial_edge_feats = latent_edge_feats
        initial_node_feats = latent_node_feats

        # During training, the feature vectors that the MPNetwork outputs for the  last self.num_class_steps message
        # passing steps are classified in order to compute the loss.
        first_class_step = self.num_enc_steps - self.num_class_steps + 1
        first_attention_step = self.num_enc_steps - self.num_attention_steps + 1 
        if self.use_attention:
            if self.graph_pruning: 
                outputs_dict = {'classified_edges': [],'att_coefficients':[],'mask':[]}
            else: outputs_dict = {'classified_edges': [],'att_coefficients':[]}
        else: 
            if self.graph_pruning:  
                outputs_dict = {'classified_edges': [],'mask':[]}
            else: outputs_dict = {'classified_edges': []}
        

        if self.graph_pruning:
            mask = torch.full((edge_index.shape[1],), True, dtype=torch.bool)
            prune_keeping_ratio = 1.0
            if self.prune_mode in ["classifier", "similarity"]:
                edge_id = edge_index.detach().transpose(0, 1).contiguous().cpu()
                row = edge_id[:, 0]
                col = edge_id[:, 1]

                row0, ind = torch.sort(row)
                _, ind_back = torch.sort(ind)
                output_row, count = torch.unique_consecutive(row0, return_counts=True)
                # cum_count = torch.cumsum(count, dim=0)

        for step in range(1, self.num_enc_steps + 1):
            
            # Reattach the initially encoded embeddings before the update
            if self.reattach_initial_edges:
                latent_edge_feats = torch.cat((initial_edge_feats, latent_edge_feats), dim=1)              # [M,16]+[M,16] -> [M,32]
            if self.reattach_initial_nodes:
                latent_node_feats = torch.cat((initial_node_feats, latent_node_feats), dim=1)

            # Message Passing Step
            if self.use_attention:
                if self.graph_pruning and torch.all(mask):
                    edge_feats = torch.zeros(latent_edge_feats.shape[0], self.edge_mlp_output_dim).cuda()
                    a = torch.zeros(self.attention_head_num, latent_edge_feats.shape[0]).cuda()

                    if self.network_split:
                        latent_node_feats, edge_feats[mask], a_mask = self.MPNet[step - 1](latent_node_feats,
                                                                                   edge_index.T[mask].T,
                                                                                   latent_edge_feats[mask])
                    else:
                        latent_node_feats, edge_feats[mask], a_mask = self.MPNet(latent_node_feats, edge_index.T[mask].T,
                                                                         latent_edge_feats[mask])
                    latent_edge_feats = edge_feats
                    a[:,mask] = a_mask

                else:
                    if self.network_split:
                         latent_node_feats, latent_edge_feats,a = self.MPNet[step-1](latent_node_feats, edge_index,
                                                                             latent_edge_feats)
                    else:
                         latent_node_feats, latent_edge_feats,a = self.MPNet(latent_node_feats, edge_index, latent_edge_feats)

            elif self.graph_pruning:
                edge_feats = torch.zeros(latent_edge_feats.shape[0],self.edge_mlp_output_dim).cuda()

                if self.network_split:
                    latent_node_feats, edge_feats[mask] = self.MPNet[step-1](latent_node_feats, edge_index.T[mask].T,
                                                                     latent_edge_feats[mask])
                else:
                    latent_node_feats, edge_feats[mask] = self.MPNet(latent_node_feats, edge_index.T[mask].T, latent_edge_feats[mask])
                latent_edge_feats = edge_feats
            else:
                if self.network_split:
                    latent_node_feats, latent_edge_feats = self.MPNet[step-1](latent_node_feats, edge_index,
                                                                         latent_edge_feats)
                else:
                    latent_node_feats, latent_edge_feats = self.MPNet(latent_node_feats, edge_index, latent_edge_feats)

            if step >= first_class_step:
                # Classification Step
                logits, _ = self.classifier(latent_edge_feats)
                pruning_this_step = self.graph_pruning and step >= self.first_prune_step \
                        and (step - self.graph_pruning) % self.prune_frequency == 0

                if ~self.graph_pruning:
                    outputs_dict['classified_edges'].append(torch.sigmoid(logits))
                elif ~pruning_this_step:
                    probabilities[mask] = torch.sigmoid(logits.view(-1)[mask])
                    outputs_dict['classified_edges'].append(probabilities)



                if self.use_attention and self.use_att_regu and step >= first_attention_step:
                    outputs_dict['att_coefficients'].append(a.clone())
            
                if pruning_this_step:

                    probabilities = torch.zeros_like(logits.view(-1))
                    probabilities[mask] = torch.sigmoid(logits.view(-1)[mask])
                    prune_keeping_ratio *= self.prune_factor

                    if self.prune_mode == "classifier naive":
                        valid_pro = probabilities[mask]
                        topk_mask = torch.full((valid_pro.shape[0],), True,dtype=torch.bool)
                        _,indice = torch.topk(valid_pro,int(len(valid_pro)*self.prune_factor),largest=False)
                        #_,indice = torch.topk(valid_pro, min(int(len(probabilities) * self.prune_factor),len(valid_pro) - N), largest=False)
                        topk_mask[indice]= False
                        mask[mask==True]=topk_mask
                        outputs_dict['mask'].append(mask.clone())


                    elif self.prune_mode == "classifier node wise":

                        """
                        probabilities = probabilities.cpu()
                        # edge_id = edge_id[ind]
                        probabilities_convert = probabilities[ind]
                        mask_convert = mask[ind]

                        # N = x.size(0)
                        for i in range(len(output_row)):

                            if i == 0:
                                prob_i = probabilities_convert[range(cum_count[i])]
                                index_i = torch.LongTensor(list(range(cum_count[i])))
                                mask_i = mask_convert[range(cum_count[i])]
                                # temp = 0
                            else:
                                prob_i = probabilities_convert[range(cum_count[i - 1], cum_count[i])]
                                index_i = torch.LongTensor(list(range(cum_count[i - 1], cum_count[i])))
                                mask_i = mask_convert[range(cum_count[i - 1], cum_count[i])]
                                # temp = cum_count[i - 1]
                            # print(index_i.shape[0])
                            # print(prob_i.shape[0])

                            index_i = index_i[mask_i]
                            prob_i = prob_i[mask_i]

                            # print(prob_i)
                            if len(index_i) <= self.prune_min_edge:
                                continue
                            discard_num = int(len(prob_i) * self.prune_factor)
                            _, ind_i = torch.topk(prob_i, discard_num, largest=False)
                            mask_convert[index_i[ind_i.long()]] = False
                            

                        mask = mask_convert[ind_back]
                        outputs_dict['mask'].append(mask.clone())
                        """
                        valid_pro = probabilities[mask]
                        valid_idx = edge_index[0][mask]

                        topk_mask = torch.ones(len(valid_pro), dtype=torch.bool)
                        valid_pro_copy = valid_pro.clone()

                        k = torch.ones(len(valid_idx)).cuda()
                        k = torch.max(scatter_add(k, valid_idx))
                        k = int(k * self.prune_factor)
                        for i in range(k):
                            _, argmin = torch_scatter.scatter_min(valid_pro_copy, valid_idx)
                            neighbor = scatter_add(topk_mask.long().cuda(), valid_idx)
                            argmin = argmin[neighbor > self.prune_min_edge]
                            topk_mask[argmin] = False
                            valid_pro_copy[argmin] = 2

                        mask[mask == True] = topk_mask
                        outputs_dict['mask'].append(mask.clone())

                    elif self.prune_mode == "similarity naive":

                        distance = torch.norm(
                            latent_node_feats[edge_index[0, :]] - latent_node_feats[edge_index[1, :]],
                            dim=1).detach()

                        valid_dis = distance[mask]
                        topk_mask = torch.full((valid_dis.shape[0],), True, dtype=torch.bool)
                        _,indice = torch.topk(valid_dis,int(len(valid_dis)*self.prune_factor),largest=False)
                        #_, indice = torch.topk(valid_dis,
                        #                       min(int(len(probabilities) * self.prune_factor), len(valid_dis) - N),
                        #                       largest=True)
                        topk_mask[indice] = False
                        mask[mask == True] = topk_mask
                        outputs_dict['mask'].append(mask.clone())


                    elif self.prune_mode == "similarity node wise":

                        distance = torch.norm(
                            latent_node_feats[edge_index[0, :]] - latent_node_feats[edge_index[1, :]],
                            dim=1).detach()

                        valid_dis = distance[mask]
                        valid_idx = edge_index[0][mask]

                        topk_mask = torch.ones(len(valid_dis), dtype=torch.bool)
                        valid_dis_copy = valid_dis.clone()
                        valid_dis_max = torch.max(valid_dis_copy)

                        k = torch.ones(len(valid_idx)).cuda()
                        k = torch.max(scatter_add(k, valid_idx))
                        k = int(k * self.prune_factor)
                        for i in range(k):
                            _, argmin = torch_scatter.scatter_min(valid_dis_copy, valid_idx)
                            neighbor = scatter_add(topk_mask.long().cuda(), valid_idx)
                            argmin = argmin[neighbor > self.prune_min_edge]
                            topk_mask[argmin] = False
                            valid_dis_copy[argmin] = valid_dis_max + 1

                        mask[mask == True] = topk_mask
                        outputs_dict['mask'].append(mask.clone())

                        """
                        distance = torch.norm(
                            latent_node_feats[edge_index[0, edge_i]] - latent_node_feats[edge_index[1, edge_i]],
                            dim=1)
                        # edge_id = edge_id[ind]
                        distance_convert = distance[ind]
                        mask_convert = mask[ind]

                        # N = x.size(0)
                        for i in range(len(output_row)):

                            if i == 0:
                                dist_i = distance_convert[range(cum_count[i])]
                                index_i = torch.LongTensor(list(range(cum_count[i])))
                                mask_i = mask_convert[range(cum_count[i])]
                                # temp = 0
                            else:
                                dist_i = distance_convert[range(cum_count[i - 1], cum_count[i])]
                                index_i = torch.LongTensor(list(range(cum_count[i - 1], cum_count[i])))
                                mask_i = mask_convert[range(cum_count[i - 1], cum_count[i])]
                                # temp = cum_count[i - 1]
                            # print(index_i.shape[0])
                            # print(prob_i.shape[0])

                            index_i = index_i[mask_i]
                            dist_i = dist_i[mask_i]

                            # print(prob_i)
                            if len(index_i) <= self.prune_min_edge:
                                continue
                            discard_num = int(len(dist_i) * self.prune_factor)
                            _, ind_i = torch.topk(dist_i, discard_num, largest=False)
                            mask_convert[index_i[ind_i.long()]] = False

                        mask = mask_convert[ind_back]
                        outputs_dict['mask'].append(mask.clone())
                        """


                    probabilities = torch.zeros_like(logits.view(-1))
                    probabilities[mask] = torch.sigmoid(logits.view(-1)[mask])
                    outputs_dict['classified_edges'].append(probabilities)

        if self.num_enc_steps == 0:
            dec_edge_feats, _ = torch.sigmoid(self.classifier(latent_edge_feats))
            outputs_dict['classified_edges'].append(dec_edge_feats)

        return outputs_dict
