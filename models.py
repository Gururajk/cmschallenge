import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pytorch_lightning as pl

'''
TODO: Implement CMSData and other data fetching functions. CMSData in
      particular should return an iterable of dictionaries, each with the
      following key-value structure:

      'patient' -> patient_dict, which
        - may only contain time-independent fields as keys
        - should ideally have all the time-independent fields as keys
      'history' -> history, a list of dictionaries, each of which
        - must contain the ‘time’ and ‘icd’ keys and otherwise
        - may contain only time-dependent fields as keys
        - should ideally have all time-dependent keys as fields'''
from cmschallenge.data import CMSData


def convert_time(time_str):
    # TODO: Implement
    pass


def hot_encode_data(data_dict, possible_values):

    enc_vec = []
    for field in sorted(list(possible_values.keys())):
        vals = possible_values[field]
        # TODO: Come up with better way to impute missing data
        if vals is None:
            enc_vec.append(data_dict.get(field, 0))
        else:
            impute = field not in data_dict
            vec_append = [1/len(vals) if impute else 0 for val in vals]
            if not impute:
                vec_append[vals[data_dict[field]]] = 1
            enc_vec += vec_append
    return enc_vec


def nn_encode(x, layers):
    vals = x
    for layer in layers:
        vals = torch.relu(layer(input.view(vals.size(0), -1)))
    return vals


class DeepDiagnoser(pl.LightningModule):

    def __init__(self, static_metadata, dynamic_metadata, icd_codes,
                 disease_progression_network):
        '''Constructor for the Deep Diagnoser Module

        Args:
            *_metadata: a dictionary mapping metadata fields to lists
                        containing all possible values for those fields; the
                        provided value should be None for non-categorical fields
                - static_metadata: describes all time-independent metadata
                - dynamic_metadata: describes all time-dependent metadata

            icd_codes: a list of all ICD codes over which the Diagnoser is
                       expected to reason and predict

            disease_progression_network: a dictionary mapping pairs of ICD
                                         codes to weights'''

        super(DeepDiagnoser, self).__init__()

        # Set up data

        # Set up static data field values
        self.stat_dat = dict()
        self.stat_inp_count = 0
        for field, vals in static_metadata.items():
            if vals is None:
                self.stat_dat[field] = None
                self.stat_inp_count += 1
            else:
                self.stat_dat[field] = {val: i for i, val in enumerate(vals)}
                self.stat_inp_count += len(vals)

        # Set up all possible ICD codes
        self.icd_codes = {code: i for i, code in enumerate(icd_codes)}

        # Set up dynamic data field values
        self.dyn_dat = dict()
        self.dyn_inp_count = len(icd_codes) + 1  # The extra 1 is for 'time'
        for field, vals in dynamic_metadata.items():
            if vals is None:
                self.dyn_dat[field] = None
                self.dyn_inp_count += 1
            else:
                self.dyn_dat[field] = {val: i for i, val in enumerate(vals)}
                self.dyn_inp_count += len(vals)

        # Set up DPN
        self.dpn = dict()
        for (u, v), w in disease_progression_network.items():
            u, v = self.icd_codes[u], self.icd_codes[v]
            self.dpn.setdefault(u, dict())[v] = w

        # Set up model parameters

        # Static Encoder
        SHD, SHW, SOW = 3, 10, 5  # num/width of hidden layers, output layer width
        self.stat_enc = [torch.nn.Linear(self.stat_inp_count, SHW)] + \
            [torch.nn.Linear(SHW, SHW) for i in range(SHD)] + \
            [torch.nn.Linear(SHW, SOW)]

        # Dynamic Encoder
        DHD, DHW, DOW = 3, 10, 5  # num/width of hidden layers, output layer width
        self.dyn_enc = [torch.nn.Linear(self.dyn_inp_count, DHW)] + \
            [torch.nn.Linear(DHW, DHW) for i in range(DHD)] + \
            [torch.nn.Linear(DHW, DOW)]

        # Sequence model
        self.sequence_model = torch.nn.LSTM(DOW, len(icd_codes))

        # DPN Processing Parameters
        self.dpn_bias = torch.rand(len(icd_codes)).type(torch.FloatTensor)
        self.dpn_bias = self.dpn_bias/self.dpn_bias.sum()
        self.dpn_bias = Variable(self.dpn_bias, requires_grad=True)

        self.dpn_pw = torch.ones(len(icd_codes), len(icd_codes))
        self.dpn_pw = self.dpn_pw.type(torch.FloatTensor)
        self.dpn_pw = Variable(self.dpn_pw, requires_grad=True)

    def tensorize_data(self, datum):

        patient_vector = hot_encode_data(datum['patient'], self.stat_dat)
        patient_vector = torch.FloatTensor(patient_vector)
        history = []
        for elem in datum['history']:
            enc_elem = [0 for i in range(len(self.icd_codes))]
            enc_elem[self.icd_codes[elem['icd']]] = 1
            enc_elem = [convert_time(elem['time'])] + enc_elem
            enc_elem += hot_encode_data(elem, self.dyn_dat)
            history.append(torch.FloatTensor(enc_elem))
        return patient_vector, history

    def forward_sequence(self, patient, history):

        hidden = nn_encode(patient, self.stat_enc)
        hidden = (hidden, hidden)
        outs = []
        for h in history:
            out, hidden = self.sequence_model(h.view(1, 1, -1), hidden)
            outs.append(out)
        return outs

    def forward_dpn(self, time_seq, icd_seq):

        N = len(self.icd_codes)
        outs = []
        diagnoses = dict()
        for t, d in zip(time_seq, icd_seq):
            basevals = self.dpn_bias
            for k, tk in diagnoses:
                dt = t - tk
                alpha = torch.zeros(N)
                for i, w in self.dpn[k].items():
                    alpha[i] = w * np.exp(-dt/w)
                basevals = basevals + alpha * self.dpn_pw[k]
            outs.append(F.sigmoid(basevals))
            diagnoses[d] = t
        return outs[1:]

    def training_step(self, batch, batch_idx):

        patient, history = self.tensorize_data(batch)
        icd_seq = [self.icd_codes[elem['icd']] for elem in batch['history']]
        time_seq = [elem[0] for elem in history]
        dpn_results = torch.cat(self.forward_dpn(time_seq, icd_seq))
        sequence_results = torch.cat(self.forward_sequence(patient, history))
        loss = F.cross_entropy(dpn_results * sequence_results,
                               torch.Tensor(icd_seq))
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(CMSData(train=True))

    @pl.data_loader
    def val_dataloader(self):
        return self.train_dataloader()

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(CMSData(train=False))
