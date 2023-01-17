import os

import torch as th
import torch.nn as nn
import pytorch_pretrained_bert as Bert
import numpy as np

from Utils.utils import load_state_dict


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, config, feature_dict=None):
        super(BertEmbeddings, self).__init__()

        self.feature_dict = feature_dict

        if feature_dict['word']:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
            nn.init.xavier_uniform_(self.word_embeddings.weight)

        if feature_dict['seg']:
            self.segment_embeddings = nn.Embedding(2, config.hidden_size)
            nn.init.xavier_uniform_(self.segment_embeddings.weight)

        if feature_dict['age']:
            self.age_embeddings = nn.Embedding(120, config.hidden_size)
            nn.init.xavier_uniform_(self.age_embeddings.weight)

        if feature_dict['gender']:
            self.gender_embeddings = nn.Embedding(2, config.hidden_size)
            nn.init.xavier_uniform_(self.gender_embeddings.weight)

        if feature_dict['position']:
            self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size). \
            from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))

        self.LayerNorm = Bert.modeling.BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, word_ids, seg_ids, posi_ids, age_ids, gender_ids):
        embeddings = self.word_embeddings(word_ids)

        if self.feature_dict['position']:
            posi_embeddings = self.posi_embeddings(posi_ids)
            embeddings = embeddings + posi_embeddings

        if self.feature_dict['age']:
            age_embeddings = self.age_embeddings(age_ids)
            embeddings = embeddings + age_embeddings

        if self.feature_dict['gender']:
            gender_embeddings = self.gender_embeddings(gender_ids)
            embeddings = embeddings + gender_embeddings

        if self.feature_dict['seg']:
            segment_embed = self.segment_embeddings(seg_ids)
            embeddings = embeddings + segment_embed

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def _init_posi_embedding(self, max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return th.tensor(lookup_table)


class BertModel(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config, feature_dict):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config=config, feature_dict=feature_dict)
        self.encoder = Bert.modeling.BertEncoder(config=config)
        self.pooler = Bert.modeling.BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, seg_ids, posi_ids, age_ids, gender_ids, attention_mask,
                output_all_encoded_layers=True):

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, seg_ids, posi_ids, age_ids, gender_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForMultiLabelPrediction(Bert.modeling.BertPreTrainedModel):
    def __init__(self, args, bert_conf, feature_dict, cls_conf, class_weights=None):
        super(BertForMultiLabelPrediction, self).__init__(bert_conf)
        self.task = cls_conf['task']
        self.bert = BertModel(bert_conf, feature_dict)
        self.dropout = nn.Dropout(bert_conf.hidden_dropout_prob)

        if self.task in ['binary', 'm30']:
            self.los_binary = nn.Linear(bert_conf.hidden_size, 1)
        elif self.task == 'real':
            self.los_real = nn.Linear(bert_conf.hidden_size, 2)  # Output = (μ, ln(σ))
        elif self.task == 'category':
            self.los_category = nn.Linear(bert_conf.hidden_size, len(cls_conf['cats']) + 1)
            nn.init.xavier_uniform_(self.los_category.weight)
            nn.init.zeros_(self.los_category.bias)

        self.apply(self.init_bert_weights)
        self.is_loss_set = False
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()
        self.va_loss = self.va_loss
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids, posi_ids=None, age_ids=None, gender_ids=None, seg_ids=None, targets=None, attention_mask=None):
        # Bert part of the model
        _, pooled_output = self.bert(input_ids, seg_ids, posi_ids, age_ids, gender_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.dropout(pooled_output)

        loss, out = None, None
        # Classification heads part of model
        if self.task in ['binary', 'm30']:
            out = self.sigmoid(self.los_binary(logits))  # BCELoss
            loss = self.bce_loss(out.squeeze(), targets.squeeze())
        elif self.task == 'real':
            out = self.los_real(logits)  # variance attenuation loss https://arxiv.org/abs/2204.09308
            loss = self.va_loss(out.squeeze(), targets.squeeze())
        elif self.task == 'category':
            out = self.los_category(logits)  # CrossEntropyLoss()
            loss = self.ce_loss(out, targets.squeeze())
        else:
            exit(f'Task: {self.task} not implemented for BertModel')

        return loss, out

    def va_loss(self, y_true, y_pred):  # variance attenuation
        mu = y_pred[:, :1]  # first output neuron
        log_sig = y_pred[:, 1:]  # second output neuron
        sig = th.exp(log_sig)  # undo the log

        return th.mean(2 * log_sig + ((y_true - mu) / sig) ** 2)
