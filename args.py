import argparse
from email.policy import default


def _add_common_args(arg_parser):
    arg_parser.add_argument('--config', type=str)

    # Input
    arg_parser.add_argument('--types_path', type=str, help="Path to type specifications")
    # Preprocessing
    arg_parser.add_argument('--tokenizer_path', type=str, help="Path to tokenizer")
    arg_parser.add_argument('--lowercase', action='store_true', default=False,
                            help="If true, input is lowercased during preprocessing")
    arg_parser.add_argument('--sampling_processes', type=int, default=4,
                            help="Number of sampling processes. 0 = no multiprocessing for sampling")
    # Logging
    arg_parser.add_argument('--label', type=str, help="Label of run. Used as the directory name of logs/models")
    arg_parser.add_argument('--log_path', type=str, help="Path do directory where training/evaluation logs are stored")
    arg_parser.add_argument('--store_examples', action='store_true',
                            help="If true, store evaluation examples on disc (in log directory)")
    arg_parser.add_argument('--example_count', type=int, default=None,
                            help="Count of evaluation example to store (if store_examples == True)")
    arg_parser.add_argument('--debug', action='store_true', default=False, help="Debugging mode on/off")

    # Model / Training / Evaluation
    arg_parser.add_argument('--model_path', type=str, help="Path to directory that contains model checkpoints")
    arg_parser.add_argument('--model_type', type=str, default="spert", help="Type of model")
    arg_parser.add_argument('--cpu', action='store_true', default=False,
                            help="If true, train/evaluate on CPU even if a CUDA device is available")
    arg_parser.add_argument('--eval_batch_size', type=int, default=1, help="Evaluation batch size")
    arg_parser.add_argument('--encoder_hidden', type=int, default=100, help="Dimensionality of encoder hidden embedding")
    arg_parser.add_argument('--kernel_size', type=int, default=3, help="Kernel size")  
    arg_parser.add_argument('--conv_layers', type=int, default=2, help="Number of CNN layers")
    arg_parser.add_argument('--prop_drop', type=float, default=0.1, help="Probability of dropout used throughout the model")
    arg_parser.add_argument('--bert_layer', type=int, default=12, help="Bert layer to use (0 for embedding, 12 for the last layer)")
    arg_parser.add_argument('--freeze_transformer', action='store_true', default=False, help="Freeze BERT weights")

    # Misc
    arg_parser.add_argument('--seed', type=int, default=None, help="Seed")
    arg_parser.add_argument('--cache_path', type=str, default=None,
                            help="Path to cache transformer models (for HuggingFace transformers library)")


def train_argparser():
    arg_parser = argparse.ArgumentParser()

    # Input
    arg_parser.add_argument('--train_path', type=str, help="Path to train dataset")
    arg_parser.add_argument('--valid_path', type=str, help="Path to validation dataset")

    # Logging
    arg_parser.add_argument('--save_path', type=str, help="Path to directory where model checkpoints are stored")
    arg_parser.add_argument('--save_best', action='store_true', default=False, help= "Save the model with best validation performance")
    arg_parser.add_argument('--init_eval', action='store_true', default=False,
                            help="If true, evaluate validation set before training")
    arg_parser.add_argument('--save_optimizer', action='store_true', default=False,
                            help="Save optimizer alongside model")
    arg_parser.add_argument('--train_log_iter', type=int, default=1, help="Log training process every x iterations")
    arg_parser.add_argument('--final_eval', action='store_true', default=False,
                            help="Evaluate the model only after training, not at every epoch")

    # Model / Training
    arg_parser.add_argument('--train_batch_size', type=int, default=2, help="Training batch size")
    arg_parser.add_argument('--epochs', type=int, default=20, help="Number of epochs")
    arg_parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    arg_parser.add_argument('--lr_bert', type=float, default=5e-5, help="Learning rate")
    arg_parser.add_argument('--lr_warmup', type=float, default=0.1,
                            help="Proportion of total train iterations to warmup in linear increase/decrease schedule")
    arg_parser.add_argument('--scheduler', type=str, default='linear_warmup', help="LR scheduler type")
    arg_parser.add_argument('--num_cycles', type=float, default= 3.0, help="Number of cycles for LR scheduler")
    arg_parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay to apply")
    arg_parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm")

    _add_common_args(arg_parser)

    return arg_parser


def eval_argparser():
    arg_parser = argparse.ArgumentParser()

    # Input
    arg_parser.add_argument('--dataset_path', type=str, help="Path to dataset")

    _add_common_args(arg_parser)

    return arg_parser
