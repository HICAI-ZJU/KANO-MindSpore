
import numpy as np
from mindspore.common.dtype import List
import mindspore as ms
import mindspore.nn as nn
import pandas as pd
import os
from argparse import Namespace
from logging import Logger
import pickle
import csv

from chemprop_ms.train.evaluate import evaluate, evaluate_predictions
from chemprop_ms.train.predict import predict
from chemprop_ms.train.train import train
from chemprop_ms.data import StandardScaler
from chemprop_ms.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop_ms.models import build_model, build_pretrain_model
from chemprop_ms.models.prompt_generator import add_functional_prompt
from chemprop_ms.nn_utils import param_count
from chemprop_ms.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func


def run_training(args: Namespace, prompt: bool, logger: Logger = None):
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    global writer
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    ms.set_context(device_target="CPU", device_id=args.gpu, mode=ms.PYNATIVE_MODE)

    # Print args
    # =============================================================================
    #     debug(pformat(vars(args)))
    # =============================================================================

    # Get data
    info('Loading data')
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    info(f'Number of tasks = {args.num_tasks}')

    # Split data
    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path, args=args, features_path=args.separate_test_features_path,
                             logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path, args=args, features_path=args.separate_val_features_path,
                            logger=logger)

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0),
                                              seed=args.seed, args=args, logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0),
                                             seed=args.seed, args=args, logger=logger)
    else:
        print('=' * 100)
        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes,
                                                     seed=args.seed, args=args, logger=logger)

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.save_smiles_splits:
        with open(args.data_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

            lines_by_smiles = {}
            indices_by_smiles = {}
            for i, line in enumerate(reader):
                smiles = line[0]
                lines_by_smiles[smiles] = line
                indices_by_smiles[smiles] = i

        all_split_indices = []
        for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
            with open(os.path.join('./train_data', name + '_smiles.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['smiles'])
                for smiles in dataset.smiles():
                    writer.writerow([smiles])
            with open(os.path.join('./train_data', name + '_full.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for smiles in dataset.smiles():
                    writer.writerow(lines_by_smiles[smiles])
            split_indices = []
            for smiles in dataset.smiles():
                split_indices.append(indices_by_smiles[smiles])
                split_indices = sorted(split_indices)
            all_split_indices.append(split_indices)
        with open(os.path.join('./train_data', 'split_indices.pckl'), 'wb') as f:
            pickle.dump(all_split_indices, f)

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)

    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)

    else:
        scaler = None

    # Get loss and metric functions
    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.dataset_type == 'multiclass':
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))


    # df_ensemble = pd.DataFrame(columns=['avg_ensemble_test_score'])
    # df_ensemble.to_csv("./metrics/ensemble_test_mae_1.csv", index=False)
    # df_validation_avg_mae = pd.DataFrame(columns=['avg_val_score', 'n_iter'])
    # df_validation_avg_mae.to_csv("./metrics/validation_avg_mae_1.csv", index=False)
    # df_validation_mae = pd.DataFrame(columns=['val_score', 'n_iter'])
    # df_validation_mae.to_csv("./metrics/validation_mae_1.csv", index=False)
    # df_test_avg_mae = pd.DataFrame(columns=['avg_test_score'])
    # df_test_avg_mae.to_csv("./metrics/test_avg_mae_1.csv", index=False)
    # df_test_mae = pd.DataFrame(columns=['test_score', 'n_iter'])
    # df_test_mae.to_csv("./metrics/test_mae_1.csv", index=False)


    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join("./train_data/", f'model_{model_idx}')
        # makedirs(save_dir)
        # # try:
        # #     writer = SummaryWriter(log_dir=save_dir)
        # # except:
        # #     writer = SummaryWriter(logdir=save_dir)
        # Load/build model
        if args.checkpoint_path is not None:
            debug(f'Loading model from {args.checkpoint_path}')

            model = build_model(args, encoder_name=args.encoder_name)
            #model.encoder.load_state_dict(ms.load_checkpoint(args.checkpoint_path), strict=False)
            param_dict = ms.load_checkpoint(args.checkpoint_path)
            ms.load_param_into_net(model.encoder, param_dict, strict_load=False)
        else:
            debug(f'Building model {model_idx}')
            model = build_model(args, encoder_name=args.encoder_name)

        if args.step == 'functional_prompt':
            add_functional_prompt(model, args)

        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        ms.save_checkpoint(model, './dumped/model.ckpt')

        # Optimizers
        # optimizer = build_optimizer(model, args)
        #
        # # Learning rate schedulers
        #scheduler = build_lr_scheduler(optimizer, args)
        scheduler = nn.exponential_decay_lr(learning_rate=args.init_lr, decay_rate=0.99, total_step=6, decay_epoch=1,
                                            step_per_epoch=args.train_data_size // args.batch_size)
        optimizer = build_optimizer(model, scheduler, args)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        for epoch in range(args.epochs):
            info(f'Epoch {epoch}')

            n_iter = train(
                model=model,
                prompt=prompt,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
            )
            val_scores = evaluate(
                model=model,
                prompt=prompt,
                data=val_data,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger
            )

            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            info(f'Validation {args.metric} = {avg_val_score:.6f}')
            # with open(os.path.join(save_dir), 'w') as f:
            #     writer = csv.writer(f)
            #     writer.writerow([f'validation_{args.metric}', avg_val_score, n_iter])
            # list_score = [avg_val_score, n_iter]
            # data_loss = pd.DataFrame([list_score])
            # data_loss.to_csv("./metrics/validation_avg_mae_1.csv", mode='a', header=False, index=False)

            test_preds = predict(
                model=model,
                prompt=prompt,
                data=test_data,
                batch_size=args.batch_size,
                scaler=scaler
            )
            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                logger=logger
            )

            # Average test score
            avg_test_score = np.nanmean(test_scores)
            info(f'test {args.metric} = {avg_test_score:.6f}')

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
                    # with open(os.path.join(save_dir), 'w') as f:
                    #     writer = csv.writer(f)
                    #     writer.writerow([f'validation_{task_name}_{args.metric}', val_score, n_iter])
                    # list_score = [val_score, n_iter]
                    # data_loss = pd.DataFrame([list_score])
                    # data_loss.to_csv("./metrics/validation_mae_1.csv", mode='a', header=False, index=False)

            # Save model checkpoint if improved validation score
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                ms.save_checkpoint(model, './dumped/model.ckpt')

                # Evaluate on test set using model with best validation score
        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        param_dict = ms.load_checkpoint('./dumped/model.ckpt')
        ms.load_param_into_net(model, param_dict)

        test_preds = predict(
            model=model,
            prompt=prompt,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler
        )
        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            logger=logger
        )
        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds)
        # Average test score
        avg_test_score = np.nanmean(test_scores)
        info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f}')
        # with open(os.path.join(save_dir), 'w') as f:
        #     # writer = csv.writer(f)
        #     # writer.writerow([f'test_{args.metric}', avg_test_score, 0])
        # list_score = [avg_test_score]
        # data_loss = pd.DataFrame([list_score])
        # data_loss.to_csv("./metrics/test_avg_mae_1.csv", mode='a', header=False, index=False)


        if args.show_individual_scores:
            # Individual test scores
            for task_name, test_score in zip(args.task_names, test_scores):
                info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')
                # with open(os.path.join(save_dir), 'w') as f:
                #     writer = csv.writer(f)
                #     writer.writerow([f'test_{task_name}_{args.metric}', test_score, n_iter])
                # list_score = [test_score,n_iter]
                # data_loss = pd.DataFrame([list_score])
                # data_loss.to_csv("./metrics/test_mae_1.csv", mode='a', header=False, index=False)

    # Evaluate ensemble on test set
    avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

    ensemble_scores = evaluate_predictions(
        preds=avg_test_preds,
        targets=test_targets,
        num_tasks=args.num_tasks,
        metric_func=metric_func,
        dataset_type=args.dataset_type,
        logger=logger
    )

    # Average ensemble score
    avg_ensemble_test_score = np.nanmean(ensemble_scores)
    info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')
    # save_dir = os.path.join("./train_data/")
    # with open(os.path.join(save_dir), 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow([f'ensemble_test_{args.metric}', avg_ensemble_test_score, 0])
    # list_score = [avg_ensemble_test_score]
    # data_score = pd.DataFrame([list_score])
    # data_score.to_csv("./metrics/ensemble_test_mae_1.csv", mode='a', header=False, index=False)

    # Individual ensemble scores
    if args.show_individual_scores:
        for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
            info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')

    return ensemble_scores
