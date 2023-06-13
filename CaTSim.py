"""CaTSim class and runner."""

import glob
import torch
import random
import numpy as np
from tqdm import tqdm, trange
from torch_geometric.nn import GCNConv 
from torch_geometric.datasets import GEDDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.utils import degree
from torch_geometric.transforms import OneHotDegree
from layers import AttentionModule, TenorNetworkModule
from utils import process_pair, calculate_loss, calculate_normalized_ged, top_k, calculate_ranking_correlation, calculate_prec_at_k
from scipy.stats import spearmanr, kendalltau

import torch.nn.functional as F


class CaTSim(torch.nn.Module):
    """
    CaTSim: Cross Contrast and Top Nodes Similarity Learning and Computation
    
    """
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(CaTSim, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.model == 0:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        elif self.args.model == 1:
            self.feature_count = self.args.tensor_neurons * 3 + self.args.bins
        elif self.args.model == 2:
            self.feature_count = self.args.tensor_neurons + self.args.bins + 2
        elif self.args.model == 3:
            self.feature_count = self.args.tensor_neurons * 3 + self.args.bins + 2
        

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist/torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_3(features, edge_index)
        return features

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)
        
        hist = self.calculate_histogram(abstract_features_1,
                                        torch.t(abstract_features_2))
        
        if self.args.model == 0: #The baseline model

            pooled_features_1 = self.attention(abstract_features_1)
            pooled_features_2 = self.attention(abstract_features_2)
            scores = self.tensor_network(pooled_features_1, pooled_features_2)
            scores = torch.t(scores)
            scores = torch.cat((scores, hist), dim=1).view(1, -1)
            scores = torch.nn.functional.relu(self.fully_connected_first(scores))
            score = torch.sigmoid(self.scoring_layer(scores))
        
        elif self.args.model == 1: #The approach 1

            x_abstract_chunks1 = torch.chunk(abstract_features_1, abstract_features_1.size()[0], dim=0)
            # transform (N,32) to N*(1,32)
            x_abstract_chunks1 = [torch.transpose(chunk, 0, 1) for chunk in x_abstract_chunks1]
            # transform to（32，1）
            x_abstract_chunks2 = torch.chunk(abstract_features_2, abstract_features_2.size()[0], dim=0)
            x_abstract_chunks2 = [torch.transpose(chunk, 0, 1) for chunk in x_abstract_chunks2]
            ## node -- node
            hist = self.calculate_histogram(abstract_features_1,
                                            torch.t(abstract_features_2))

            ## graph -- graph
            pooled_features_1 = self.attention(abstract_features_1)
            pooled_features_2 = self.attention(abstract_features_2)
            scores = self.tensor_network(pooled_features_1, pooled_features_2)
            scores = torch.t(scores)

            ## node 1 -- graph 2
            scores_sum1 =  []
            for i in range(abstract_features_1.size()[0]):
                x_i = x_abstract_chunks1[i]
                scores_i = self.tensor_network(x_i, pooled_features_2)
                scores_sum1.append(scores_i) 
            stacked_tensor_1 = torch.stack(scores_sum1)

            scores_avg_1 = torch.mean(stacked_tensor_1, dim=0)
            scores2 = torch.t(scores_avg_1)

            ## node 2 -- graph 1
            scores_sum2 =  []
            for i in range(abstract_features_2.size()[0]):
                x_i = x_abstract_chunks2[i]
                scores_i = self.tensor_network(x_i, pooled_features_1)
                scores_sum2.append(scores_i) 
            stacked_tensor_2 = torch.stack(scores_sum2)

            scores_avg_2 = torch.mean(stacked_tensor_2, dim=0)
            #scores_avg = scores_sum / abstract_features_1.size()[0]
            scores3 = torch.t(scores_avg_2)

            ## concatenated 

            scores = torch.cat((scores, scores2, scores3, hist), dim=0).view(1, -1)# (16,)


            scores = torch.nn.functional.relu(self.fully_connected_first(scores))
            score = torch.sigmoid(self.scoring_layer(scores))
        
        elif self.args.model == 2: #The approach 2
            
            if len(features_1) > 2 and len(features_2) > 2:
                nodes_1 = edge_index_1[0].tolist()
                nodes_2 = edge_index_2[0].tolist()

                count_1 = [0] * len(features_1)
                count_2 = [0] * len(features_2)

                for i in range(len(count_1)):
                    count_1[i] = nodes_1.count(i)
                for i in range(len(count_2)):
                    count_2[i] = nodes_2.count(i)

                toptwo_1 = top_k(count_1, 2)
                toptwo_2 = top_k(count_2, 2)

                cos1 = F.cosine_similarity(abstract_features_1[toptwo_1[0]], abstract_features_2[toptwo_2[0]], dim=0)
                cos2 = F.cosine_similarity(abstract_features_1[toptwo_1[1]], abstract_features_2[toptwo_2[1]], dim=0)
                temp = torch.tensor([[cos1, cos2]])
            else:
                temp = torch.tensor([[0, 0]])



        # step2: attention produce graph embedding
            pooled_features_1 = self.attention(abstract_features_1)
            pooled_features_2 = self.attention(abstract_features_2)  # 32*1

        # step3: NTN produce Similarity Score
            scores = self.tensor_network(pooled_features_1, pooled_features_2)  # 16*1
            scores = torch.t(scores)  # 1*16

            scores = torch.cat((scores, temp), dim=1)  # 1*18

            scores = torch.cat((scores, hist), dim=1).view(1, -1)

            temp = self.fully_connected_first(scores)  # 1*16
            scores = torch.nn.functional.relu(temp)

            score = torch.sigmoid(self.scoring_layer(scores))  # tensor([[0.5316]], grad_fn=<SigmoidBackward0>)

        elif self.args.model == 3: #The final model CaTSim

            x_abstract_chunks1 = torch.chunk(abstract_features_1, abstract_features_1.size()[0], dim=0)
            x_abstract_chunks1 = [torch.transpose(chunk, 0, 1) for chunk in x_abstract_chunks1]

            x_abstract_chunks2 = torch.chunk(abstract_features_2, abstract_features_2.size()[0], dim=0)
            x_abstract_chunks2 = [torch.transpose(chunk, 0, 1) for chunk in x_abstract_chunks2]
        
            hist = self.calculate_histogram(abstract_features_1,
                                            torch.t(abstract_features_2))

            if len(features_1) > 2 and len(features_2) > 2:
                nodes_1 = edge_index_1[0].tolist()
                nodes_2 = edge_index_2[0].tolist()

                count_1 = [0] * len(features_1)
                count_2 = [0] * len(features_2)

                for i in range(len(count_1)):
                    count_1[i] = nodes_1.count(i)
                for i in range(len(count_2)):
                    count_2[i] = nodes_2.count(i)

                toptwo_1 = top_k(count_1, 2)
                toptwo_2 = top_k(count_2, 2)

                cos1 = F.cosine_similarity(abstract_features_1[toptwo_1[0]], abstract_features_2[toptwo_2[0]], dim=0)
                cos2 = F.cosine_similarity(abstract_features_1[toptwo_1[1]], abstract_features_2[toptwo_2[1]], dim=0)
                temp = torch.tensor([[cos1, cos2]])
            else:
                temp = torch.tensor([[0, 0]])


            pooled_features_1 = self.attention(abstract_features_1)
            pooled_features_2 = self.attention(abstract_features_2)
            scores = self.tensor_network(pooled_features_1, pooled_features_2)
            scores = torch.t(scores)

            scores_sum1 =  []
            for i in range(abstract_features_1.size()[0]):
                x_i = x_abstract_chunks1[i]
                scores_i = self.tensor_network(x_i, pooled_features_2)
                scores_sum1.append(scores_i) 
            stacked_tensor_1 = torch.stack(scores_sum1)

            scores_avg_1 = torch.mean(stacked_tensor_1, dim=0)
            scores2 = torch.t(scores_avg_1)
            scores_sum2 =  []
            for i in range(abstract_features_2.size()[0]):
                x_i = x_abstract_chunks2[i]
                scores_i = self.tensor_network(x_i, pooled_features_1)
                scores_sum2.append(scores_i) 
            stacked_tensor_2 = torch.stack(scores_sum2)

            scores_avg_2 = torch.mean(stacked_tensor_2, dim=0)
            scores3 = torch.t(scores_avg_2)         

        ## concatenated 
                  
            scores = torch.cat((scores, scores2, scores3, hist), dim=0).view(1, -1)# (16,)
            scores = torch.cat((scores, temp), dim=1)
            
            scores = torch.nn.functional.relu(self.fully_connected_first(scores))
            score = torch.sigmoid(self.scoring_layer(scores))

        
        return score

class CaTSimTrainer(object):
    """
    CaTSim model trainer.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """

        print("\nPreparing dataset.\n")
        self.args = args
        data_dir = './GSC_datasets'
        self.dataset = self.args.dataset
        # AIDS700nef, LINUX, IMDBMulti or ALKANE

        self.training_graphs = GEDDataset(data_dir + '/{}'.format(self.dataset), self.dataset, train=True) # 560
        self.testing_graphs = GEDDataset(data_dir + '/{}'.format(self.dataset), self.dataset, train=False) # 140
        if self.dataset == "ALKANE":  # len = 150, but only 0-119 has valid value
            self.testing_graphs = self.training_graphs[96:] # len = 96
            self.training_graphs = self.training_graphs[0:96] # len = 24

        self.nged_matrix = self.training_graphs.norm_ged
        self.ged_matrix = self.training_graphs.ged

        if self.training_graphs[0].x is None:
            max_degree = 0
            for g in self.training_graphs + self.testing_graphs:
                if g.edge_index.size(1) > 0:
                    max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
            self.one_hot_degree = OneHotDegree(max_degree, cat=False)
            self.training_graphs.transform = self.one_hot_degree
            self.testing_graphs.transform = self.one_hot_degree
            
    
        self.setup_model()


    def setup_model(self):
        """
        Creating a CaTSim.
        """
        self.model = CaTSim(self.args, self.training_graphs.num_features)


    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: Zipped loaders as list.
        """
        # 5 batches in a dataloader
        source_loader = DataLoader(self.training_graphs.shuffle(),
                               batch_size=self.args.batch_size)

        target_loader = DataLoader(self.training_graphs.shuffle(),
                               batch_size=self.args.batch_size)

        return list(zip(source_loader, target_loader)) # randomly select pair of graph


    def process_batch(self,data): # data is batch pair

        self.optimizer.zero_grad()

        my_loss = 0

        for i in range(len(data[0])):  #
            temp = {}
            temp["edge_index_1"] = data[0][i]['edge_index']
            temp["edge_index_2"] = data[1][i]['edge_index']
            temp["features_1"] = data[0][i]['x']
            temp["features_2"] = data[1][i]['x']
            nGED = self.nged_matrix[data[0][i]['i'], data[1][i]['i']]
            temp["target"] = torch.exp(-nGED)

            prediction = self.model(temp)[0] # 2d to 1d

            my_loss = my_loss + torch.nn.functional.mse_loss(prediction, temp["target"])

        my_loss.backward(retain_graph=True)
        self.optimizer.step()
        return my_loss.item()

    def fit(self):
        """
        Fitting a model.
        """
        print("\nModel training.\n")

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)

        self.model.train()
        epochs = trange(self.args.epochs, leave=True, desc="Epoch")
        for epoch in epochs:
            batches = self.create_batches()
            self.loss_sum = 0
            main_index = 0
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
                loss_score = self.process_batch(batch)
                main_index = main_index + len(batch[0])
                self.loss_sum = self.loss_sum + loss_score 
                loss = self.loss_sum/main_index
                epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
        


    def score_batch(self, data): # data is batch pair

        my_loss = 0
        x = np.array([])

        for i in range(len(data[0])):  # len(data[0])
            temp = {}
            temp["edge_index_1"] = data[0][i]['edge_index']
            temp["edge_index_2"] = data[1][i]['edge_index']
            temp["features_1"] = data[0][i]['x']
            temp["features_2"] = data[1][i]['x']

            nGED = self.nged_matrix[data[0][i]['i'], data[1][i]['i']]
            temp["target"] = torch.exp(-nGED)

            prediction = self.model(temp)[0] # 2d to 1d
            x = np.append(x, prediction.detach().numpy())
            my_loss = my_loss + torch.nn.functional.mse_loss(prediction, temp["target"])

        return my_loss.item(), x
    
    def transform(self, data):
        """
        Getting ged for graph pair and grouping with data into dictionary.
        :param data: Graph pair.
        :return new_data: Dictionary with data.
        """
        new_data = dict()
        
        new_data["g1"] = data[0]
        new_data["g2"] = data[1]
        
        normalized_ged = self.nged_matrix[data[0]["i"].reshape(-1).tolist(), data[1]["i"].reshape(-1).tolist()].tolist()
        
        new_data["target"] = torch.from_numpy(np.exp([(-el) for el in normalized_ged])).view(-1).float()
        
        ged = self.ged_matrix[data[0]["i"].reshape(-1).tolist(), data[1]["i"].reshape(-1).tolist()].tolist()
        new_data["target_ged"] = torch.from_numpy(np.array([(el) for el in ged])).view(-1).float()
        
        return new_data


    def score(self):
        """
        Scoring on the test set.
        """
        

        print("\n\nModel evaluation.\n")
        self.model.eval()
        loss_sum = 0
        
        ground_truth = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        ground_truth_ged = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        prediction_mat = np.empty((len(self.testing_graphs), len(self.training_graphs)))
		
        rho_list = []
        tau_list = []
        prec_at_10_list = []
        prec_at_20_list = []
        
        t = tqdm(total=len(self.testing_graphs) * len(self.training_graphs))  # 140 * 560 = 78400


        for i, g in enumerate(self.testing_graphs):
            source_batch = Batch.from_data_list([g] * len(self.training_graphs))
            target_batch = Batch.from_data_list(self.training_graphs)
            
            data = self.transform((source_batch, target_batch))
            target = data["target"]  # len = 560, 560 similarity scores
            ground_truth[i] = target
            target_ged = data["target_ged"]  # 560 GED
            ground_truth_ged[i] = target_ged
            
            loss, prediction_mat[i] = self.score_batch((source_batch, target_batch))
       
            loss_sum = loss_sum + loss
            
            rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
            tau_list.append(calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]))
            prec_at_10_list.append(calculate_prec_at_k(10, prediction_mat[i], ground_truth[i], ground_truth_ged[i]))
            prec_at_20_list.append(calculate_prec_at_k(20, prediction_mat[i], ground_truth[i], ground_truth_ged[i]))

            t.update(len(self.training_graphs))


        self.model_error = loss_sum / (len(self.testing_graphs) * len(self.training_graphs))
        
        rho = np.mean(rho_list).item()
        tau = np.mean(tau_list).item()
        prec_at_10 = np.mean(prec_at_10_list).item()
        prec_at_20 = np.mean(prec_at_20_list).item()

        print("\nmse(10^-3): " + str(round(self.model_error * 1000, 5)) + ".")
        
        print("Spearman's rho: " + str(round(rho, 5)) + ".")
        print("Kendall's tau: " + str(round(tau, 5)) + ".")
        print("p@10: " + str(round(prec_at_10, 5)) + ".")
        print("p@20: " + str(round(prec_at_20, 5)) + ".")

        if self.args.load == None:
            self.save()

        # 6.06683 * 10^-3， 50 epochs

    def save(self):
        PATH = './src/model_saved/Model'+str(self.args.model)+'_'+self.args.dataset+'_'+str(round(self.model_error * 1000, 5))+'.pth'
        torch.save(self.model, PATH)

    def load(self):
        self.args.dataset = self.args.load.split("_")[2]
        self.dataset = self.args.dataset
        
        data_dir = './GSC_datasets'
        # AIDS700nef, LINUX, IMDBMulti or ALKANE

        self.training_graphs = GEDDataset(data_dir + '/{}'.format(self.dataset), self.dataset, train=True) # 560
        self.testing_graphs = GEDDataset(data_dir + '/{}'.format(self.dataset), self.dataset, train=False) # 140
        if self.dataset == "ALKANE":  # len = 150, but only 0-119 has valid value
            self.testing_graphs = self.training_graphs[96:] # len = 96
            self.training_graphs = self.training_graphs[0:96] # len = 24

        self.nged_matrix = self.training_graphs.norm_ged
        self.ged_matrix = self.training_graphs.ged

        if self.training_graphs[0].x is None:
            max_degree = 0
            for g in self.training_graphs + self.testing_graphs:
                if g.edge_index.size(1) > 0:
                    max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
            self.one_hot_degree = OneHotDegree(max_degree, cat=False)
            self.training_graphs.transform = self.one_hot_degree
            self.testing_graphs.transform = self.one_hot_degree
        
        self.model = torch.load(self.args.load)