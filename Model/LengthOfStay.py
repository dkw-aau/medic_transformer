import torch
import torch.nn as nn
import pytorch_pretrained_bert as Bert
import numpy as np


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, config, feature_dict=None):
        super(BertEmbeddings, self).__init__()

        self.feature_dict = feature_dict

        if feature_dict['word']:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        if feature_dict['position']:
            self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size). \
            from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))

        self.LayerNorm = Bert.modeling.BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, word_ids, posi_ids):
        embeddings = self.word_embeddings(word_ids)

        if self.feature_dict['position']:
            posi_embeddings = self.posi_embeddings(posi_ids)
            embeddings = embeddings + posi_embeddings

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

        return torch.tensor(lookup_table)


class BertModel(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config, feature_dict):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config=config, feature_dict=feature_dict)
        self.encoder = Bert.modeling.BertEncoder(config=config)
        self.pooler = Bert.modeling.BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, posi_ids, attention_mask,
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

        embedding_output = self.embeddings(input_ids, posi_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForMultiLabelPrediction(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config, feature_dict, cls_heads, cls_config=None):
        super(BertForMultiLabelPrediction, self).__init__(config)
        self.cls_heads = cls_heads
        self.bert = BertModel(config, feature_dict)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if 'los_binary' in cls_heads:
            self.los_binary = nn.Linear(config.hidden_size, 1)
        if 'los_real' in cls_heads:
            self.los_real = nn.Linear(config.hidden_size, 1)
        if 'los_binned' in cls_heads:
            self.los_binned = nn.Linear(config.hidden_size, cls_config['bins'])
        if 'req_hosp' in cls_heads:
            self.req_hosp = nn.Linear(config.hidden_size, 1)
        if 'mortality_30' in cls_heads:
            self.mortality_30 = nn.Linear(config.hidden_size, 1)

        self.apply(self.init_bert_weights)
        self.is_loss_set = False
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, posi_ids=None, targets=None, attention_mask=None):
        # Bert part of the model
        _, pooled_output = self.bert(input_ids, posi_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.dropout(pooled_output)

        # Classification heads part of model
        outputs = {}
        loss = None
        self.is_loss_set = False
        if 'los_binary' in self.cls_heads:
            out = self.sigmoid(self.los_binary(logits))  # BCELoss
            loss = self.add_loss(self.bce_loss(out.view(-1, 1), targets['los'].view(-1, 1)), loss)
            outputs['los'] = out
        if 'los_real' in self.cls_heads:
            out = self.los_real(logits)  # nn.L1Loss()
            loss = self.add_loss(self.l1_loss(out.view(-1, 1), targets['los'].view(-1, 1)), loss)
            outputs['los'] = out
        if 'los_binned' in self.cls_heads:
            out = self.los_binned(logits)  # CrossEntropyLoss()
            loss = self.add_loss(self.ce_loss(out.view(-1, 1), targets['los'].view(-1, 1)), loss)
            outputs['los'] = out
        if 'req_hosp' in self.cls_heads:
            out = self.sigmoid(self.req_hosp(logits))  # BCELoss
            loss = self.add_loss(self.bce_loss(out.view(-1, 1), targets['hosp'].view(-1, 1)), loss)
            outputs['hosp'] = out
        if 'mortality_30' in self.cls_heads:
            out = self.sigmoid(self.mortality_30(logits))  # BCELoss
            loss = self.add_loss(self.bce_loss(out.view(-1, 1), targets['m30'].view(-1, 1)), loss)
            outputs['m30'] = out

        return loss, outputs

    def add_loss(self, new_loss, prev_loss=None):
        if self.is_loss_set:
            prev_loss += new_loss
            return prev_loss
        else:
            self.is_loss_set = True
            return new_loss
