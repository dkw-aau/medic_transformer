
def get_file_config():
    return {
        'path': r'\\srvsas9402\Platform_AN\3_Projekt_1\Innovationsprojektet\Data',
        'corpus': 'corpus_small',  # vocabulary and data
        'output_path': '../Outputs',  # where to save model
        'pretrain_name': 'MLM_Model.pt',  # model name
        'finetune_name': 'Best_Model.pt',  # output model name
        'log_file': 'log.txt',  # log path
    }


def get_global_params():
    return {
        'gradient_accumulation_steps': 1,
        'step_eval': 5
    }


def get_optim_config():
    return {
        'lr': 3e-5,
        'warmup_proportion': 0.1,
        'weight_decay': 0.01
    }


def get_train_params():
    return {
        'batch_size': 256,
        'use_cuda': False,
        'max_len_seq': 256,
        'device': 'cpu'  # 'cuda:0'
    }


def get_model_config(vocab, train_params):
    return {
        'vocab_size': len(vocab['token2index'].keys()),  # num embeddings
        'hidden_size': 288,  # word embedding and index embedding hidden size
        'max_position_embedding': train_params['max_len_seq'],  # maximum number of tokens
        'hidden_dropout_prob': 0.1,  # dropout rate
        'num_hidden_layers': 2,  # number of multi-head attention layers required
        'num_attention_heads': 6,  # number of attention heads
        'attention_probs_dropout_prob': 0.1,  # multi-head attention dropout rate
        'intermediate_size': 512,  # the size of the "intermediate" layer in the transformer encoder
        'hidden_act': 'gelu',
        # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
        'initializer_range': 0.02,  # parameter weight initializer range
    }
