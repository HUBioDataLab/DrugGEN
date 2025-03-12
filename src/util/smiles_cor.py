import os
import time
import random
import re
import itertools
import statistics

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.data import TabularDataset, Field, BucketIterator, Iterator

from rdkit import Chem, rdBase, RDLogger
from rdkit.Chem import (
    MolStandardize,
    GraphDescriptors,
    Lipinski,
    AllChem,
)
from rdkit.Chem.rdSLNParse import MolFromSLN
from rdkit.Chem.rdmolfiles import MolFromSmiles
from chembl_structure_pipeline import standardizer

RDLogger.DisableLog('rdApp.*')

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

##################################################################################################
##################################################################################################
#                                                                                                #
#  THIS SCRIPT IS DIRECTLY ADAPTED FROM https://github.com/LindeSchoenmaker/SMILES-corrector     #
#                                                                                                #
##################################################################################################
##################################################################################################
def is_smiles(array,
              TRG,
              reverse: bool,
              return_output=False,
              src=None,
              src_field=None):
    """Turns predicted tokens within batch into smiles and evaluates their validity
    Arguments:
        array: Tensor with most probable token for each location for each sequence in batch
            [trg len, batch size]
        TRG: target field for getting tokens from vocab
        reverse (bool): True if the target sequence is reversed
        return_output (bool): True if output sequences and their validity should be saved
    Returns:
        df: dataframe with correct and incorrect sequences
        valids: list with booleans that show if prediction was a valid SMILES (True) or invalid one (False)
        smiless: list of the predicted smiles
    """
    trg_field = TRG
    valids = []
    smiless = []
    if return_output:
        df = pd.DataFrame()
    else:
        df = None
    batch_size = array.size(1)
    # check if the first token should be removed, first token is zero because
    # outputs initaliazed to all be zeros
    if int((array[0, 0]).tolist()) == 0:
        start = 1
    else:
        start = 0
    # for each sequence in the batch
    for i in range(0, batch_size):
        # turns sequence from tensor to list skipps first row as this is not
        # filled in in forward
        sequence = (array[start:, i]).tolist()
        # goes from embedded to tokens
        trg_tokens = [trg_field.vocab.itos[int(t)] for t in sequence]
        # print(trg_tokens)
        # takes all tokens untill eos token, model would be faster if did this
        # one step earlier, but then changes in vocab order would disrupt.
        rev_tokens = list(
            itertools.takewhile(lambda x: x != "<eos>", trg_tokens))
        if reverse:
            rev_tokens = rev_tokens[::-1]
        smiles = "".join(rev_tokens)
        # determine how many valid smiles are made
        valid = True if MolFromSmiles(smiles) else False
        valids.append(valid)
        smiless.append(smiles)
        if return_output:
            if valid:
                df.loc[i, "CORRECT"] = smiles
            else:
                df.loc[i, "INCORRECT"] = smiles

    # add the original drugex outputs to the _de dataframe
    if return_output and src is not None:
        for i in range(0, batch_size):
            # turns sequence from tensor to list skipps first row as this is
            # <sos> for src
            sequence = (src[1:, i]).tolist()
            # goes from embedded to tokens
            src_tokens = [src_field.vocab.itos[int(t)] for t in sequence]
            # takes all tokens untill eos token, model would be faster if did
            # this one step earlier, but then changes in vocab order would
            # disrupt.
            rev_tokens = list(
                itertools.takewhile(lambda x: x != "<eos>", src_tokens))
            smiles = "".join(rev_tokens)
            df.loc[i, "ORIGINAL"] = smiles

    return df, valids, smiless


def is_unchanged(array,
                 TRG,
                 reverse: bool,
                 return_output=False,
                 src=None,
                 src_field=None):
    """Checks is output is different from input
    Arguments:
        array: Tensor with most probable token for each location for each sequence in batch
            [trg len, batch size]
        TRG: target field for getting tokens from vocab
        reverse (bool): True if the target sequence is reversed
        return_output (bool): True if output sequences and their validity should be saved
    Returns:
        df: dataframe with correct and incorrect sequences
        valids: list with booleans that show if prediction was a valid SMILES (True) or invalid one (False)
        smiless: list of the predicted smiles
    """
    trg_field = TRG
    sources = []
    batch_size = array.size(1)
    unchanged = 0

    # check if the first token should be removed, first token is zero because
    # outputs initaliazed to all be zeros
    if int((array[0, 0]).tolist()) == 0:
        start = 1
    else:
        start = 0

    for i in range(0, batch_size):
        # turns sequence from tensor to list skipps first row as this is <sos>
        # for src
        sequence = (src[1:, i]).tolist()
        # goes from embedded to tokens
        src_tokens = [src_field.vocab.itos[int(t)] for t in sequence]
        # takes all tokens untill eos token, model would be faster if did this
        # one step earlier, but then changes in vocab order would disrupt.
        rev_tokens = list(
            itertools.takewhile(lambda x: x != "<eos>", src_tokens))
        smiles = "".join(rev_tokens)
        sources.append(smiles)

    # for each sequence in the batch
    for i in range(0, batch_size):
        # turns sequence from tensor to list skipps first row as this is not
        # filled in in forward
        sequence = (array[start:, i]).tolist()
        # goes from embedded to tokens
        trg_tokens = [trg_field.vocab.itos[int(t)] for t in sequence]
        # print(trg_tokens)
        # takes all tokens untill eos token, model would be faster if did this
        # one step earlier, but then changes in vocab order would disrupt.
        rev_tokens = list(
            itertools.takewhile(lambda x: x != "<eos>", trg_tokens))
        if reverse:
            rev_tokens = rev_tokens[::-1]
        smiles = "".join(rev_tokens)
        # determine how many valid smiles are made
        valid = True if MolFromSmiles(smiles) else False
        if not valid:
            if smiles == sources[i]:
                unchanged += 1

    return unchanged


def molecule_reconstruction(array, TRG, reverse: bool, outputs):
    """Turns target tokens within batch into smiles and compares them to predicted output smiles
    Arguments:
        array: Tensor with target's token for each location for each sequence in batch
            [trg len, batch size]
        TRG: target field for getting tokens from vocab
        reverse (bool): True if the target sequence is reversed
        outputs: list of predicted SMILES sequences
    Returns:
         matches(int): number of total right molecules
    """
    trg_field = TRG
    matches = 0
    targets = []
    batch_size = array.size(1)
    # for each sequence in the batch
    for i in range(0, batch_size):
        # turns sequence from tensor to list skipps first row as this is not
        # filled in in forward
        sequence = (array[1:, i]).tolist()
        # goes from embedded to tokens
        trg_tokens = [trg_field.vocab.itos[int(t)] for t in sequence]
        # takes all tokens untill eos token, model would be faster if did this
        # one step earlier, but then changes in vocab order would disrupt.
        rev_tokens = list(
            itertools.takewhile(lambda x: x != "<eos>", trg_tokens))
        if reverse:
            rev_tokens = rev_tokens[::-1]
        smiles = "".join(rev_tokens)
        targets.append(smiles)
    for i in range(0, batch_size):
        m = MolFromSmiles(targets[i])
        p = MolFromSmiles(outputs[i])
        if p is not None:
            if m.HasSubstructMatch(p) and p.HasSubstructMatch(m):
                matches += 1
    return matches


def complexity_whitlock(mol: Chem.Mol, includeAllDescs=False):
    """
    Complexity as defined in DOI:10.1021/jo9814546
    S: complexity = 4*#rings + 2*#unsat + #hetatm + 2*#chiral
    Other descriptors:
        H: size = #bonds (Hydrogen atoms included)
        G: S + H
        Ratio: S / H
    """
    mol_ = Chem.Mol(mol)
    nrings = Lipinski.RingCount(mol_) - Lipinski.NumAromaticRings(mol_)
    Chem.rdmolops.SetAromaticity(mol_)
    unsat = sum(1 for bond in mol_.GetBonds()
                if bond.GetBondTypeAsDouble() == 2)
    hetatm = len(mol_.GetSubstructMatches(Chem.MolFromSmarts("[!#6]")))
    AllChem.EmbedMolecule(mol_)
    Chem.AssignAtomChiralTagsFromStructure(mol_)
    chiral = len(Chem.FindMolChiralCenters(mol_))
    S = 4 * nrings + 2 * unsat + hetatm + 2 * chiral
    if not includeAllDescs:
        return S
    Chem.rdmolops.Kekulize(mol_)
    mol_ = Chem.AddHs(mol_)
    H = sum(bond.GetBondTypeAsDouble() for bond in mol_.GetBonds())
    G = S + H
    R = S / H
    return {"WhitlockS": S, "WhitlockH": H, "WhitlockG": G, "WhitlockRatio": R}


def complexity_baronechanon(mol: Chem.Mol):
    """
    Complexity as defined in DOI:10.1021/ci000145p
    """
    mol_ = Chem.Mol(mol)
    Chem.Kekulize(mol_)
    Chem.RemoveStereochemistry(mol_)
    mol_ = Chem.RemoveHs(mol_, updateExplicitCount=True)
    degree, counts = 0, 0
    for atom in mol_.GetAtoms():
        degree += 3 * 2**(atom.GetExplicitValence() - atom.GetNumExplicitHs() -
                          1)
        counts += 3 if atom.GetSymbol() == "C" else 6
    ringterm = sum(map(lambda x: 6 * len(x), mol_.GetRingInfo().AtomRings()))
    return degree + counts + ringterm


def calc_complexity(array,
                    TRG,
                    reverse,
                    valids,
                    complexity_function=GraphDescriptors.BertzCT):
    """Calculates the complexity of inputs that are not correct.
    Arguments:
        array: Tensor with target's token for each location for each sequence in batch
            [trg len, batch size]
        TRG: target field for getting tokens from vocab
        reverse (bool): True if the target sequence is reversed
        valids: list with booleans that show if prediction was a valid SMILES (True) or invalid one (False)
        complexity_function: the type of complexity measure that will be used
            GraphDescriptors.BertzCT
            complexity_whitlock
            complexity_baronechanon
    Returns:
         matches(int): mean of complexity values
    """
    trg_field = TRG
    sources = []
    complexities = []
    loc = torch.BoolTensor(valids)
    # only keeps rows in batch size dimension where valid is false
    array = array[:, loc == False]
    # should check if this still works
    # array = torch.transpose(array, 0, 1)
    array_size = array.size(1)
    for i in range(0, array_size):
        # turns sequence from tensor to list skipps first row as this is not
        # filled in in forward
        sequence = (array[1:, i]).tolist()
        # goes from embedded to tokens
        trg_tokens = [trg_field.vocab.itos[int(t)] for t in sequence]
        # takes all tokens untill eos token, model would be faster if did this
        # one step earlier, but then changes in vocab order would disrupt.
        rev_tokens = list(
            itertools.takewhile(lambda x: x != "<eos>", trg_tokens))
        if reverse:
            rev_tokens = rev_tokens[::-1]
        smiles = "".join(rev_tokens)
        sources.append(smiles)
    for source in sources:
        try:
            m = MolFromSmiles(source)
        except BaseException:
            m = MolFromSLN(source)
        complexities.append(complexity_function(m))
    if len(complexities) > 0:
        mean = statistics.mean(complexities)
    else:
        mean = 0
    return mean


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class Convo:
    """Class for training and evaluating transformer and convolutional neural network
    
    Methods
    -------
    train_model()
        train model for initialized number of epochs
    evaluate(return_output)
        use model with validation loader (& optionally drugex loader) to get test loss & other metrics
    translate(loader)
        translate inputs from loader (different from evaluate in that no target sequence is used)
    """

    def train_model(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        log = open(f"{self.out}.log", "a")
        best_error = np.inf
        for epoch in range(self.epochs):
            self.train()
            start_time = time.time()
            loss_train = 0
            for i, batch in enumerate(self.loader_train):
                optimizer.zero_grad()
                # changed src,trg call to match with bentrevett
                # src, trg = batch['src'], batch['trg']
                trg = batch.trg
                src = batch.src
                output, attention = self(src, trg[:, :-1])
                # feed the source and target into def forward to get the output
                # Xuhan uses forward for this, with istrain = true
                output_dim = output.shape[-1]
                # changed
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                # output = output[:,:,0]#.view(-1)
                # output = output[1:].view(-1, output.shape[-1])
                # trg = trg[1:].view(-1)
                loss = nn.CrossEntropyLoss(
                    ignore_index=self.TRG.vocab.stoi[self.TRG.pad_token])
                a, b = output.view(-1), trg.to(self.device).view(-1)
                # changed
                # loss = loss(output.view(0), trg.view(0).to(device))
                loss = loss(output, trg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
                optimizer.step()
                loss_train += loss.item()
                # turned off for now, as not using voc so won't work, output is a tensor
                # output = [(trg len - 1) * batch size, output dim]
                # smiles, valid = is_valid_smiles(output, reversed)
                # if valid:
                #    valids += 1
                #    smiless.append(smiles)
            # added .dataset becaue len(iterator) gives len(self.dataset) /
            # self.batch_size)
            loss_train /= len(self.loader_train)
            info = f"Epoch: {epoch+1:02} step: {i} loss_train: {loss_train:.4g}"
            # model is used to generate trg based on src from the validation set to assess performance
            # similar to Xuhan, although he doesn't use the if loop
            if self.loader_valid is not None:
                return_output = False
                if epoch + 1 == self.epochs:
                    return_output = True
                (
                    valids,
                    loss_valid,
                    valids_de,
                    df_output,
                    df_output_de,
                    right_molecules,
                    complexity,
                    unchanged,
                    unchanged_de,
                ) = self.evaluate(return_output)
                reconstruction_error = 1 - right_molecules / len(
                    self.loader_valid.dataset)
                error = 1 - valids / len(self.loader_valid.dataset)
                complexity = complexity / len(self.loader_valid)
                unchan = unchanged / (len(self.loader_valid.dataset) - valids)
                info += f" loss_valid: {loss_valid:.4g} error_rate: {error:.4g} molecule_reconstruction_error_rate: {reconstruction_error:.4g} unchanged: {unchan:.4g} invalid_target_complexity: {complexity:.4g}"
                if self.loader_drugex is not None:
                    error_de = 1 - valids_de / len(self.loader_drugex.dataset)
                    unchan_de = unchanged_de / (
                        len(self.loader_drugex.dataset) - valids_de)
                    info += f" error_rate_drugex: {error_de:.4g} unchanged_drugex: {unchan_de:.4g}"

                if reconstruction_error < best_error:
                    torch.save(self.state_dict(), f"{self.out}.pkg")
                    best_error = reconstruction_error
                    last_save = epoch
                else:
                    if epoch - last_save >= 10 and best_error != 1:
                        torch.save(self.state_dict(), f"{self.out}_last.pkg")
                        (
                            valids,
                            loss_valid,
                            valids_de,
                            df_output,
                            df_output_de,
                            right_molecules,
                            complexity,
                            unchanged,
                            unchanged_de,
                        ) = self.evaluate(True)
                        end_time = time.time()
                        epoch_mins, epoch_secs = epoch_time(
                            start_time, end_time)
                        info += f" Time: {epoch_mins}m {epoch_secs}s"
                 
                        break
            elif error < best_error:
                torch.save(self.state_dict(), f"{self.out}.pkg")
                best_error = error
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            info += f" Time: {epoch_mins}m {epoch_secs}s"
   
        
        torch.save(self.state_dict(), f"{self.out}_last.pkg")
        log.close()
        self.load_state_dict(torch.load(f"{self.out}.pkg"))
        df_output.to_csv(f"{self.out}.csv", index=False)
        df_output_de.to_csv(f"{self.out}_de.csv", index=False)

    def evaluate(self, return_output):
        self.eval()
        test_loss = 0
        df_output = pd.DataFrame()
        df_output_de = pd.DataFrame()
        valids = 0
        valids_de = 0
        unchanged = 0
        unchanged_de = 0
        right_molecules = 0
        complexity = 0
        with torch.no_grad():
            for _, batch in enumerate(self.loader_valid):
                trg = batch.trg
                src = batch.src
                output, attention = self.forward(src, trg[:, :-1])
                pred_token = output.argmax(2)
                array = torch.transpose(pred_token, 0, 1)
                trg_trans = torch.transpose(trg, 0, 1)
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                src_trans = torch.transpose(src, 0, 1)
                df_batch, valid, smiless = is_smiles(
                    array, self.TRG, reverse=True, return_output=return_output)
                unchanged += is_unchanged(
                    array,
                    self.TRG,
                    reverse=True,
                    return_output=return_output,
                    src=src_trans,
                    src_field=self.SRC,
                )
                matches = molecule_reconstruction(trg_trans,
                                                  self.TRG,
                                                  reverse=True,
                                                  outputs=smiless)
                complexity += calc_complexity(trg_trans,
                                              self.TRG,
                                              reverse=True,
                                              valids=valid)
                if df_batch is not None:
                    df_output = pd.concat([df_output, df_batch],
                                          ignore_index=True)
                right_molecules += matches
                valids += sum(valid)
                # trg = trg[1:].view(-1)
                # output, trg = output[1:].view(-1, output.shape[-1]), trg[1:].view(-1)
                loss = nn.CrossEntropyLoss(
                    ignore_index=self.TRG.vocab.stoi[self.TRG.pad_token])
                loss = loss(output, trg)
            test_loss += loss.item()
            if self.loader_drugex is not None:
                for _, batch in enumerate(self.loader_drugex):
                    src = batch.src
                    output = self.translate_sentence(src, self.TRG,
                                                     self.device)
                    # checks the number of valid smiles
                    pred_token = output.argmax(2)
                    array = torch.transpose(pred_token, 0, 1)
                    src_trans = torch.transpose(src, 0, 1)
                    df_batch, valid, smiless = is_smiles(
                        array,
                        self.TRG,
                        reverse=True,
                        return_output=return_output,
                        src=src_trans,
                        src_field=self.SRC,
                    )
                    unchanged_de += is_unchanged(
                        array,
                        self.TRG,
                        reverse=True,
                        return_output=return_output,
                        src=src_trans,
                        src_field=self.SRC,
                    )
                    if df_batch is not None:
                        df_output_de = pd.concat([df_output_de, df_batch],
                                                 ignore_index=True)
                    valids_de += sum(valid)
        return (
            valids,
            test_loss / len(self.loader_valid),
            valids_de,
            df_output,
            df_output_de,
            right_molecules,
            complexity,
            unchanged,
            unchanged_de,
        )

    def translate(self, loader):
        self.eval()
        df_output_de = pd.DataFrame()
        valids_de = 0
        with torch.no_grad():
            for _, batch in enumerate(loader):
                src = batch.src
                output = self.translate_sentence(src, self.TRG, self.device)
                # checks the number of valid smiles
                pred_token = output.argmax(2)
                array = torch.transpose(pred_token, 0, 1)
                src_trans = torch.transpose(src, 0, 1)
                df_batch, valid, smiless = is_smiles(
                    array,
                    self.TRG,
                    reverse=True,
                    return_output=True,
                    src=src_trans,
                    src_field=self.SRC,
                )
                if df_batch is not None:
                    df_output_de = pd.concat([df_output_de, df_batch],
                                             ignore_index=True)
                valids_de += sum(valid)
        return valids_de, df_output_de


class Encoder(nn.Module):

    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout,
                 max_length, device):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src = [batch size, src len]
        # src_mask = [batch size, src len]
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = (torch.arange(0, src_len).unsqueeze(0).repeat(batch_size,
                                                            1).to(self.device))
        # pos = [batch size, src len]
        src = self.dropout((self.tok_embedding(src) * self.scale) +
                           self.pos_embedding(pos))
        # src = [batch size, src len, hid dim]
        for layer in self.layers:
            src = layer(src, src_mask)
        # src = [batch size, src len, hid dim]
        return src


class EncoderLayer(nn.Module):

    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads,
                                                      dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]
        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]
        # positionwise feedforward
        _src = self.positionwise_feedforward(src)
        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]

        return src


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]
        Q = Q.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch size, n heads, query len, key len]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)
        # attention = [batch size, n heads, query len, key len]
        x = torch.matmul(self.dropout(attention), V)
        # x = [batch size, n heads, query len, head dim]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, query len, n heads, head dim]
        x = x.view(batch_size, -1, self.hid_dim)
        # x = [batch size, query len, hid dim]
        x = self.fc_o(x)
        # x = [batch size, query len, hid dim]
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):

    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]
        x = self.dropout(torch.relu(self.fc_1(x)))
        # x = [batch size, seq len, pf dim]
        x = self.fc_2(x)
        # x = [batch size, seq len, hid dim]

        return x


class Decoder(nn.Module):

    def __init__(
        self,
        output_dim,
        hid_dim,
        n_layers,
        n_heads,
        pf_dim,
        dropout,
        max_length,
        device,
    ):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
            for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = (torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size,
                                                            1).to(self.device))
        # pos = [batch size, trg len]
        trg = self.dropout((self.tok_embedding(trg) * self.scale) +
                           self.pos_embedding(pos))
        # trg = [batch size, trg len, hid dim]
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]
        output = self.fc_out(trg)
        # output = [batch size, trg len, output dim]
        return output, attention


class DecoderLayer(nn.Module):

    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads,
                                                      dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]
        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]
        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src,
                                                 src_mask)
        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]
        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]
        return trg, attention


class Seq2Seq(nn.Module, Convo):

    def __init__(
        self,
        encoder,
        decoder,
        src_pad_idx,
        trg_pad_idx,
        device,
        loader_train: DataLoader,
        out: str,
        loader_valid=None,
        loader_drugex=None,
        epochs=100,
        lr=0.0005,
        clip=0.1,
        reverse=True,
        TRG=None,
        SRC=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.loader_train = loader_train
        self.out = out
        self.loader_valid = loader_valid
        self.loader_drugex = loader_drugex
        self.epochs = epochs
        self.lr = lr
        self.clip = clip
        self.reverse = reverse
        self.TRG = TRG
        self.SRC = SRC

    def make_src_mask(self, src):
        # src = [batch size, src len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = [batch size, 1, 1, src len]
        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask = [batch size, 1, 1, trg len]
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(
            torch.ones((trg_len, trg_len), device=self.device)).bool()
        # trg_sub_mask = [trg len, trg len]
        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask = [batch size, 1, trg len, trg len]
        return trg_mask

    def forward(self, src, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len]
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]
        enc_src = self.encoder(src, src_mask)
        # enc_src = [batch size, src len, hid dim]
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]
        return output, attention

    def translate_sentence(self, src, trg_field, device, max_len=202):
        self.eval()
        src_mask = self.make_src_mask(src)
        with torch.no_grad():
            enc_src = self.encoder(src, src_mask)
        trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
        batch_size = src.shape[0]
        trg = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg = trg.repeat(batch_size, 1)
        for i in range(max_len):
            # turned model into self.
            trg_mask = self.make_trg_mask(trg)
            with torch.no_grad():
                output, attention = self.decoder(trg, enc_src, trg_mask,
                                                 src_mask)
            pred_tokens = output.argmax(2)[:, -1].unsqueeze(1)
            trg = torch.cat((trg, pred_tokens), 1)

        return output


def remove_floats(df: pd.DataFrame, subset: str):
    """Preprocessing step to remove any entries that are not strings"""
    df_subset = df[subset]
    df[subset] = df[subset].astype(str)
    # only keep entries that stayed the same after applying astype str
    df = df[df[subset] == df_subset].copy()

    return df


def smi_tokenizer(smi: str, reverse=False) -> list:
    """
    Tokenize a SMILES molecule
    """
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    # tokens = ['<sos>'] + [token for token in regex.findall(smi)] + ['<eos>']
    tokens = [token for token in regex.findall(smi)]
    # assert smi == ''.join(tokens[1:-1])
    assert smi == "".join(tokens[:])
    # try:
    #     assert smi == "".join(tokens[:])
    # except:
    #     print(smi)
    #     print("".join(tokens[:]))
    if reverse:
        return tokens[::-1]
    return tokens


def init_weights(m: nn.Module):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def initialize_model(folder_out: str,
                     data_source: str,
                     error_source: str,
                     device: torch.device,
                     threshold: int,
                     epochs: int,
                     layers: int = 3,
                     batch_size: int = 16,
                     invalid_type: str = "all",
                     num_errors: int = 1,
                     validation_step=False):
    """Create encoder decoder models for specified model (currently only translator) & type of invalid SMILES

    param data: collection of invalid, valid SMILES pairs
    param invalid_smiles_path: path to previously generated invalid SMILES
    param invalid_type: type of errors introduced into invalid SMILES

    return:

    """

    # set fields
    SRC = Field(
        tokenize=lambda x: smi_tokenizer(x),
        init_token="<sos>",
        eos_token="<eos>",
        batch_first=True,
    )
    TRG = Field(
        tokenize=lambda x: smi_tokenizer(x, reverse=True),
        init_token="<sos>",
        eos_token="<eos>",
        batch_first=True,
    )

    if validation_step:
        train, val = TabularDataset.splits(
            path=f'{folder_out}errors/split/',
            train=f"{data_source}_{invalid_type}_{num_errors}_errors_train.csv",
            validation=
            f"{data_source}_{invalid_type}_{num_errors}_errors_dev.csv",
            format="CSV",
            skip_header=False,
            fields={
                "ERROR": ("src", SRC),
                "STD_SMILES": ("trg", TRG)
            },
        )
        SRC.build_vocab(train, val, max_size=1000)
        TRG.build_vocab(train, val, max_size=1000)
    else:
        train = TabularDataset(
            path=
            f'{folder_out}{data_source}_{invalid_type}_{num_errors}_errors.csv',
            format="CSV",
            skip_header=False,
            fields={
                "ERROR": ("src", SRC),
                "STD_SMILES": ("trg", TRG)
            },
        )
        SRC.build_vocab(train, max_size=1000)
        TRG.build_vocab(train, max_size=1000)

    drugex = TabularDataset(
        path=error_source,
        format="csv",
        skip_header=False,
        fields={
            "SMILES": ("src", SRC),
            "SMILES_TARGET": ("trg", TRG)
        },
    )


    #SRC.vocab = torch.load('vocab_src.pth')
    #TRG.vocab = torch.load('vocab_trg.pth')

    # model parameters
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 256
    ENC_LAYERS = layers
    DEC_LAYERS = layers
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    # add 2 to length for start and stop tokens
    MAX_LENGTH = threshold + 2

    # model name
    MODEL_OUT_FOLDER = f"{folder_out}"

    MODEL_NAME = "transformer_%s_%s_%s_%s_%s" % (
        invalid_type, num_errors, data_source, BATCH_SIZE, layers)
    if not os.path.exists(MODEL_OUT_FOLDER):
        os.mkdir(MODEL_OUT_FOLDER)

    out = os.path.join(MODEL_OUT_FOLDER, MODEL_NAME)

    torch.save(SRC.vocab, f'{out}_vocab_src.pth')
    torch.save(TRG.vocab, f'{out}_vocab_trg.pth')

    # iterator is a dataloader
    # iterator to pass to the same length and create batches in which the
    # amount of padding is minimized
    if validation_step:
        train_iter, val_iter = BucketIterator.splits(
            (train, val),
            batch_sizes=(BATCH_SIZE, 256),
            sort_within_batch=True,
            shuffle=True,
            # the BucketIterator needs to be told what function it should use to
            # group the data.
            sort_key=lambda x: len(x.src),
            device=device,
        )
    else:
        train_iter = BucketIterator(
            train,
            batch_size=BATCH_SIZE,
            sort_within_batch=True,
            shuffle=True,
            # the BucketIterator needs to be told what function it should use to
            # group the data.
            sort_key=lambda x: len(x.src),
            device=device,
        )
        val_iter = None

    drugex_iter = Iterator(
        drugex,
        batch_size=64,
        device=device,
        sort=False,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        repeat=False,
    )


    # model initialization

    enc = Encoder(
        INPUT_DIM,
        HID_DIM,
        ENC_LAYERS,
        ENC_HEADS,
        ENC_PF_DIM,
        ENC_DROPOUT,
        MAX_LENGTH,
        device,
    )
    dec = Decoder(
        OUTPUT_DIM,
        HID_DIM,
        DEC_LAYERS,
        DEC_HEADS,
        DEC_PF_DIM,
        DEC_DROPOUT,
        MAX_LENGTH,
        device,
    )

    model = Seq2Seq(
        enc,
        dec,
        SRC_PAD_IDX,
        TRG_PAD_IDX,
        device,
        train_iter,
        out=out,
        loader_valid=val_iter,
        loader_drugex=drugex_iter,
        epochs=EPOCHS,
        TRG=TRG,
        SRC=SRC,
    ).to(device)




    return model, out, SRC


def train_model(model, out, assess):
    """Apply given weights (& assess performance or train further) or start training new model

    Args:
        model: initialized model
        out: .pkg file with model parameters
        asses: bool 

    Returns:
        model with (new) weights
    """

    if os.path.exists(f"{out}.pkg") and assess:


        model.load_state_dict(torch.load(f=out + ".pkg"))
        (
            valids,
            loss_valid,
            valids_de,
            df_output,
            df_output_de,
            right_molecules,
            complexity,
            unchanged,
            unchanged_de,
        ) = model.evaluate(True)


        # log = open('unchanged.log', 'a')
        # info = f'type: comb unchanged: {unchan:.4g} unchanged_drugex: {unchan_de:.4g}'
        # print(info, file=log, flush = True)
        # print(valids_de)
        # print(unchanged_de)

        # print(unchan)
        # print(unchan_de)
        # df_output_de.to_csv(f'{out}_de_new.csv', index = False)

        # error_de = 1 - valids_de / len(drugex_iter.dataset)
        # print(error_de)
        # df_output.to_csv(f'{out}_par.csv', index = False)

    elif os.path.exists(f"{out}.pkg"):

        # starts from the model after the last epoch, not the best epoch
        model.load_state_dict(torch.load(f=out + "_last.pkg"))
        # need to change how log file names epochs
        model.train_model()
    else:

        model = model.apply(init_weights)
        model.train_model()

    return model


def correct_SMILES(model, out, error_source, device, SRC):
    """Model that is given corrects SMILES and return number of correct ouputs and dataframe containing all outputs
    Args:
        model: initialized model
        out: .pkg file with model parameters
        asses: bool 

    Returns:
        valids: number of fixed outputs
        df_output: dataframe containing output (either correct or incorrect) & original input
    """
    ## account for tokens that are not yet in SRC without changing existing SRC token embeddings
    errors = TabularDataset(
        path=error_source,
        format="csv",
        skip_header=False,
        fields={"SMILES": ("src", SRC)},
    )

    errors_loader = Iterator(
        errors,
        batch_size=64,
        device=device,
        sort=False,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        repeat=False,
    )
    model.load_state_dict(torch.load(f=out + ".pkg",map_location=torch.device('cpu')))
    # add option to use different iterator maybe?

    valids, df_output = model.translate(errors_loader)
    #df_output.to_csv(f"{error_source}_fixed.csv", index=False)


    return valids, df_output



class smi_correct(object):
    def __init__(self, model_name, trans_file_path):
    # set random seed, used for error generation & initiation transformer
    
        self.SEED = 42
        random.seed(self.SEED)
        self.model_name = model_name
        self.folder_out = "data/"
        
        self.trans_file_path = trans_file_path

        if not os.path.exists(self.folder_out):
            os.makedirs(self.folder_out)
            
        self.invalid_type = 'multiple'
        self.num_errors = 12
        self.threshold = 200
        self.data_source = f"PAPYRUS_{self.threshold}"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.initialize_source = 'data/papyrus_rnn_S.csv' # change this path
        
    def standardization_pipeline(self, smile):
        desalter = MolStandardize.fragment.LargestFragmentChooser()
        std_smile = None
        if not isinstance(smile, str): return None
        m = Chem.MolFromSmiles(smile)
        # skips smiles for which no mol file could be generated
        if m is not None:
            # standardizes
            std_m = standardizer.standardize_mol(m)
            # strips salts
            std_m_p, exclude = standardizer.get_parent_mol(std_m)
            if not exclude:
                # choose largest fragment for rare cases where chembl structure
                # pipeline leaves 2 fragments
                std_m_p_d = desalter.choose(std_m_p)
                std_smile = Chem.MolToSmiles(std_m_p_d)
        return std_smile      
    
    def remove_smiles_duplicates(self, dataframe: pd.DataFrame,
                             subset: str) -> pd.DataFrame:
        return dataframe.drop_duplicates(subset=subset)  
    
    def correct(self, smi):
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, out, SRC = initialize_model(self.folder_out,
                                        self.data_source,
                                        error_source=self.initialize_source,
                                        device=device,
                                        threshold=self.threshold,
                                        epochs=30,
                                        layers=3,
                                        batch_size=16,
                                        invalid_type=self.invalid_type,
                                        num_errors=self.num_errors)

        valids, df_output = correct_SMILES(model, out, smi, device,
                                        SRC)
        
        df_output["SMILES"] = df_output.apply(lambda row: self.standardization_pipeline(row["CORRECT"]), axis=1)
        
        df_output = self.remove_smiles_duplicates(df_output, subset="SMILES").drop(columns=["CORRECT", "INCORRECT", "ORIGINAL"]).dropna()
        
        return df_output