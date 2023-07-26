import strategies
import load_data

import warnings
import argparse
warnings.filterwarnings("ignore")


def main(args):
    config = {
        "strategy_name": args.strategy,
        "model_name": args.model_name,
        "dataset_name": args.dataset,
        "num_clients":args.num_clients,
        "batch_size":args.batch_size,
        "weighted":True,
        "reset_per_round":args.reset_per_round,
        "device":args.device,
        "stepsize":args.step_size,
        "rounds": args.num_rounds,
        "local_epochs": args.num_local_epochs,
        "central_epochs": args.num_central_epochs,
        "attribute_skew": args.attribute_skew,
        "label_skew": args.label_skew,
        "label_alpha": args.label_alpha,
        "label_n": args.label_n,
        "attribute_alpha": args.attribute_alpha,
        "purity": args.purity,
        "num_quantiles": args.num_quantiles,
        "test_set_fraction": args.test_set_fraction,
        "shared_set_fraction": args.shared_set_fraction,
        "evaluation_averaging": "weighted",
    }
    # load the raw data from file
    config = load_data.load_raw_data(config)
    # apply split for train, shared, test and distribute train over clients
    config = load_data.distribute_skewed_data(config)
    # initialize learning strategy
    learning_strategy = strategies.get_strategy_by_name(config)
    # perform training
    _ = learning_strategy.run(config, filename=args.log_file, log_per_round=args.log_per_round)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run Federated Learning')
    parser.add_argument('--strategy',
                        help='choose a federated learning strategy', 
                        choices=["fedavg", "fedprox", "fedatt", "feded", "feddf", "fedmd"], 
                        default="fedavg", 
                        required=True)
    parser.add_argument('--dataset', 
                        help='name of the dataset', 
                        default="covtype", 
                        required=True)
    parser.add_argument('--label_skew', 
                        help='kind of label or quantity skew to apply', 
                        choices=[None, "homogeneous", "label_distribution", "label_quantity", "quantity", "clustering"], 
                        default=None, 
                        required=False)
    parser.add_argument('--attribute_skew', 
                        help='kind of attribute skew to apply', 
                        choices=[None, "noise", "availability"], 
                        default=None, 
                        required=False)
    parser.add_argument('--label_alpha', 
                        help='level of imbalance if label_skew is "label_distribution", "quantity"', 
                        type=float, 
                        default=2., 
                        required=False)
    parser.add_argument('--label_n', 
                        help='max number of classes available to clients if label_skew is "label_quantity" ', 
                        type=int, 
                        default=2, 
                        required=False)
    parser.add_argument('--attribute_alpha', 
                        help='level of imbalance if attribute_skew is on', 
                        type=float, 
                        default=2., 
                        required=False)
    parser.add_argument('--num_clients', 
                        help='number of clients in the cohort', 
                        type=int, 
                        default=10, 
                        required=False)
    parser.add_argument('--purity', 
                        help='purity of clutering-based label_skew', 
                        type=float, 
                        default=.5, 
                        required=False)
    parser.add_argument('--num_quantiles', 
                        help='required for regression tasks to peform apply label skew', 
                        type=int, 
                        default=4, 
                        required=False)
    parser.add_argument('--num_rounds', 
                        help='rounds of training', 
                        type=int, 
                        default=100, 
                        required=False)
    parser.add_argument('--num_local_epochs', 
                        help='number of local training epochs', 
                        type=int, 
                        default=5, 
                        required=False)
    parser.add_argument('--num_central_epochs', 
                        help='required for strategies that retain centrallyy', 
                        type=int, 
                        default=5, 
                        required=False)
    parser.add_argument('--model_name', 
                        help='choose model as defined in model.py', 
                        default="auto", 
                        required=False)
    parser.add_argument('--batch_size', 
                        help='batch size', 
                        type=int, 
                        default=64, 
                        required=False)
    parser.add_argument('--device', 
                        help='cpu or cuda', 
                        choices=["cpu", "cuda"], 
                        default="cpu", 
                        required=False)
    parser.add_argument('--test_set_fraction', 
                        help='fraction of test set', 
                        type=float, 
                        default=.2, 
                        required=False)
    parser.add_argument('--shared_set_fraction', 
                        help='fraction of the shared set requried for some strategies', 
                        type=float, 
                        default=0.001, 
                        required=False)
    parser.add_argument('--log_per_round', 
                        help='log performance per round', 
                        choices=[True, False], 
                        default=True, 
                        required=False)
    parser.add_argument('--log_file', 
                        help='file location. None writes to console', 
                        default=None, 
                        required=False)
    parser.add_argument('--step_size', 
                        help='step size', 
                        type=float,
                        default=1.2, 
                        required=False)
    parser.add_argument('--reset_per_round', 
                        help='only for model distillation strategies. If the central model should be reset on each round', 
                        choices=[True, False], 
                        default=False, 
                        required=False)

    args = parser.parse_args()

    main(args)