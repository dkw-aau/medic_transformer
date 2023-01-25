import torch as th
import torch.nn as nn
from torch.nn.functional import sigmoid
import pytorch_pretrained_bert as Bert
import numpy as np


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, config, features=None):
        super(BertEmbeddings, self).__init__()

        self.features = features

        if 'word' in features:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        if 'age' in features:
            self.age_embeddings = nn.Embedding(120, config.hidden_size)

        if 'gender' in features:
            self.gender_embeddings = nn.Embedding(2, config.hidden_size)

        if 'position' in features:
            self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size). \
            from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))

        self.LayerNorm = Bert.modeling.BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, word_ids, posi_ids, age_ids, gender_ids):
        embeddings = self.word_embeddings(word_ids)

        if 'position' in self.features:
            posi_embeddings = self.posi_embeddings(posi_ids)
            embeddings = embeddings + posi_embeddings

        if 'age' in self.features:
            age_embeddings = self.age_embeddings(age_ids)
            embeddings = embeddings + age_embeddings

        if 'gender' in self.features:
            gender_embeddings = self.gender_embeddings(gender_ids)
            embeddings = embeddings + gender_embeddings

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
    def __init__(self, config, features):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config=config, features=features)
        self.encoder = Bert.modeling.BertEncoder(config=config)
        self.pooler = Bert.modeling.BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, posi_ids, age_ids, gender_ids, attention_mask, output_all_encoded_layers=True):

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

        embedding_output = self.embeddings(input_ids, posi_ids, age_ids, gender_ids)
        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class MBERT(Bert.modeling.BertPreTrainedModel):
    def __init__(self, bert_conf, workload, features, task, scaler, num_classes):
        super(MBERT, self).__init__(bert_conf)
        self.task = task
        self.scaler = scaler
        self.num_classes = num_classes
        self.workload = workload

        self.mbert = BertModel(bert_conf, features)
        self.dropout = nn.Dropout(bert_conf.hidden_dropout_prob)
        self.scaler = scaler

        # Pretraining cls head
        if self.workload == 'mlm':
            self.cls = Bert.modeling.BertOnlyMLMHead(bert_conf, self.mbert.embeddings.word_embeddings.weight)
        elif self.task == 'binary':
            self.cls = nn.Linear(bert_conf.hidden_size, 1)
        elif self.task == 'real':
            self.cls_mean = th.nn.Linear(bert_conf.hidden_size, 1)  # mean
            self.cls_std = th.nn.Linear(bert_conf.hidden_size, 1)  # std
        elif self.task == 'category':
            self.cls = nn.Linear(bert_conf.hidden_size, num_classes)

        # Initialize weights
        self.apply(self.init_bert_weights)

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.ne_loss = self.neg_log_loss
        self.ii_ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.jitter = 1e-6

    def forward(self, input_ids, posi_ids=None, age_ids=None, gender_ids=None, targets=None, attention_mask=None):
        # Bert part of the model
        sequence_output, pooled_output = self.mbert(input_ids, posi_ids, age_ids, gender_ids, attention_mask, output_all_encoded_layers=False)

        if self.workload == 'mlm':
            preds = self.cls(sequence_output)
            loss = self.ii_ce_loss(preds.view(-1, self.config.vocab_size), targets.view(-1))
            return loss

        logits = self.dropout(pooled_output)

        # Classification heads part of model
        loss, out = None, None
        if self.task == 'binary':
            out = sigmoid(self.cls(logits))
            loss = self.bce_loss(out.squeeze(), targets.squeeze())
        elif self.task == 'category':
            out = self.cls(logits)
            loss = self.ce_loss(out, targets.squeeze())
        elif self.task == 'real':
            # Compute mean and std layers
            mean = self.cls_mean(logits)
            std = th.nn.functional.softplus(self.cls_std(logits)) + self.jitter
            out = th.distributions.Normal(mean, std)
            loss = self.ne_loss(out, targets)
        else:
            exit(f'Task: {self.task} not implemented for BertModel')

        return loss, out

    def neg_log_loss(self, preds, target):
        neg_log_likelihood = -preds.log_prob(target)
        return th.mean(neg_log_likelihood)
