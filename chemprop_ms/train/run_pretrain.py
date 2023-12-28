import numpy as np
from mindspore.common.dtype import List

from .evaluate import evaluate, evaluate_predictions
from chemprop_ms.utils import get_loss_func, get_metric_func, load_checkpoint, makedirs
import os
from argparse import Namespace
from logging import Logger

import mindspore as ms
import mindspore.nn as nn
from mindspore import ParameterTuple
from mindspore import SummaryRecord

from chemprop_ms.models import ContrastiveLoss
from chemprop_ms.torchlight import initialize_exp, snapshot
from chemprop_ms.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop_ms.models import build_model, build_pretrain_model
from chemprop_ms.models.prompt_generator import add_functional_prompt
from chemprop_ms.nn_utils import param_count
from chemprop_ms.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint
from chemprop_ms.data import MoleculeDataset


import pandas as pd



def pre_training(args: Namespace, logger: Logger = None):
    global model2, model1, celllist, model1_param, model2_param, loss
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    '''if args.gpu is not None:
        ms.context.set_context(device_target="GPU", device_id=2)'''

    # Print args
    # =============================================================================
    #     debug(pformat(vars(args)))
    # =============================================================================

    # Get data
    debug('Loading data')
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = 1
    args.data_size = len(data)

    debug(f'Total size = {len(data)}')
    #
    #
    # args.data_size = len(data)
    # debug(f'Total size = {len(data)}')


    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx}')
            # model = build_model(args)
            model1 = build_pretrain_model(args, encoder_name='CMPNN')
            model2 = build_pretrain_model(args, encoder_name='CMPNN')
            celllist = nn.CellList()
            celllist.append(model1)
            celllist.append(model2)
            model1_param = ParameterTuple(celllist[0].trainable_params())
            model2_param = ParameterTuple(celllist[1].trainable_params())

        debug(model1)
        debug(f'Number of M1 parameters = {param_count(model1):,}')
        '''if args.cuda:
            debug('Moving model to cuda')
            model1 = model1.cuda()'''

        debug(model2)
        debug(f'Number of M2 parameters = {param_count(model2):,}')

        logger, dump_folder = initialize_exp(Namespace(**args.__dict__))
        dump_folder = f'{dump_folder}-model'
        '''

        logger, dump_folder = initialize_exp(Namespace(**args.__dict__))
        dump_folder = f'{dump_folder}-model'

        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        args.device = device'''
        ms.set_context(device_target="CPU", device_id=args.gpu, mode=ms.PYNATIVE_MODE)
        # criterion = ContrastiveLoss(loss_computer='nce_softmax', temperature=args.temperature, args=args)
        # optimizer = nn.Adam([{"params": model1.trainable_params()},{"params": model2.trainable_params()}], lr=3e-5)
        # optimizer = Adam([{"params": model1.parameters()}], lr=3e-5)
        # optimizer = nn.Adam(model1.trainable_params(), learning_rate=3e-5)
        scheduler = nn.exponential_decay_lr(learning_rate=3e-4, decay_rate=0.99, total_step=6, decay_epoch=1,
                                            step_per_epoch=2)
        optimizer = nn.Adam([{"params": model1_param}, {"params": model2_param}], learning_rate=3e-5)

        mol = MoleculeDataset(data)
        smiles, features = mol.smiles(), mol.features()

        train_loader = ms.dataset.GeneratorDataset(smiles,
                                                   # batch_size=args.batch_size,
                                                   column_names=['smiles'],
                                                   shuffle=True,
                                                   num_parallel_workers=1,
                                                   )

        batch_size = 256
        train_loader = train_loader.batch(batch_size=batch_size)
        # viz = Visdom()
        # viz.line([0.], [0.], win='train_acc', opts=dict(title='train acc'))
        # viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
        df_1 = pd.DataFrame(columns=['time', 'step', 'train Loss'])
        df_1.to_csv("./train_data/train_loss.csv", index=False)


        # Run training
        def forward_fn(batch):
            criterion = ContrastiveLoss(loss_computer='nce_softmax', temperature=args.temperature, args=args)
            emb1 = model1('pretrain', False, batch, None)
            emb2 = model2('pretrain', True, batch, None)
            loss = criterion(emb1, emb2)
            return loss

        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

        # Define function of one-step training
        def train_step(batch):

            loss, grads = grad_fn(batch)
            # print(loss)
            # print(grads)
            optimizer(grads)
            return loss

        def train_loop(dataset, epochs):
            size = dataset.get_dataset_size()
            model1.set_train()
            model2.set_train()
            for batch, (data) in enumerate(dataset.create_tuple_iterator()):
                loss = train_step(data)
                print(loss)
                if batch % 1 == 0:
                    # loss, current = loss.asnumpy(), batch
                    l = loss.asnumpy()
                    print(f"loss: {l:>7f}  [{batch:>3d}/{size:>3d}]")
                with SummaryRecord('./summary_9.5_1') as summary_record:
                    current_step = (epochs - 1) * size + batch
                    # Note2: The output should be a scalar, and use 'add_value' method to record loss.
                    # Note3: You must use the 'record(step)' method to record the data of this step.
                    if current_step % 5 == 0:
                        # summary_record.add_value('scalar', 'loss', loss)
                        # summary_record.record(current_step)
                        # viz.line([loss], [current_step], win='train_acc', update='append')
                        list_loss = [current_step, loss]
                        data_loss = pd.DataFrame([list_loss])
                        data_loss.to_csv("./train_data/train_loss.csv", mode='a', header=False, index=False)

        for epoch in range(args.epochs):
            step_per_schedule = 1
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train_loop(train_loader, epoch + 1)
            # test_loop(test_loader, epoch + 1)
            # test_loop(loader,epoch + 1)
            # if (epoch + 1) % step_per_schedule == 0:
            #     scheduler.step()
            if epoch % 2 == 0:
                # ms.save_checkpoint(model1, "./model_original.ckpt")
                # ms.save_checkpoint(model2, "./model_augment.ckpt")
                ms.save_checkpoint(model1, "./ckpt/model_new1_original"+"%d.ckpt"%(epoch+1))
                ms.save_checkpoint(model2, "./ckpt/model_new1_augment"+"%d.ckpt"%(epoch+1))
        # logger.info(f'[{epoch}/{args.epochs}] train loss {loss.item():.4f}')
    # return embimport numpy as np
from mindspore.common.dtype import List

from .evaluate import evaluate, evaluate_predictions
from chemprop_ms.utils import get_loss_func, get_metric_func, load_checkpoint, makedirs
import os
from argparse import Namespace
from logging import Logger

import mindspore as ms
import mindspore.nn as nn
from mindspore import ParameterTuple
from mindspore import SummaryRecord

from chemprop_ms.models import ContrastiveLoss
from chemprop_ms.torchlight import initialize_exp, snapshot
from chemprop_ms.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop_ms.models import build_model, build_pretrain_model
from chemprop_ms.models.prompt_generator import add_functional_prompt
from chemprop_ms.nn_utils import param_count
from chemprop_ms.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint
from chemprop_ms.data import MoleculeDataset


import pandas as pd



def pre_training(args: Namespace, logger: Logger = None):
    global model2, model1, celllist, model1_param, model2_param, loss
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    '''if args.gpu is not None:
        ms.context.set_context(device_target="GPU", device_id=2)'''

    # Print args
    # =============================================================================
    #     debug(pformat(vars(args)))
    # =============================================================================

    # Get data
    debug('Loading data')
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = 1
    args.data_size = len(data)

    debug(f'Total size = {len(data)}')
    #
    #
    # args.data_size = len(data)
    # debug(f'Total size = {len(data)}')


    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx}')
            # model = build_model(args)
            model1 = build_pretrain_model(args, encoder_name='CMPNN')
            model2 = build_pretrain_model(args, encoder_name='CMPNN')
            celllist = nn.CellList()
            celllist.append(model1)
            celllist.append(model2)
            model1_param = ParameterTuple(celllist[0].trainable_params())
            model2_param = ParameterTuple(celllist[1].trainable_params())

        debug(model1)
        debug(f'Number of M1 parameters = {param_count(model1):,}')
        '''if args.cuda:
            debug('Moving model to cuda')
            model1 = model1.cuda()'''

        debug(model2)
        debug(f'Number of M2 parameters = {param_count(model2):,}')

        logger, dump_folder = initialize_exp(Namespace(**args.__dict__))
        dump_folder = f'{dump_folder}-model'
        '''

        logger, dump_folder = initialize_exp(Namespace(**args.__dict__))
        dump_folder = f'{dump_folder}-model'

        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        args.device = device'''
        ms.set_context(device_target="CPU", device_id=args.gpu, mode=ms.PYNATIVE_MODE)
        # criterion = ContrastiveLoss(loss_computer='nce_softmax', temperature=args.temperature, args=args)
        # optimizer = nn.Adam([{"params": model1.trainable_params()},{"params": model2.trainable_params()}], lr=3e-5)
        # optimizer = Adam([{"params": model1.parameters()}], lr=3e-5)
        # optimizer = nn.Adam(model1.trainable_params(), learning_rate=3e-5)
        scheduler = nn.exponential_decay_lr(learning_rate=3e-4, decay_rate=0.99, total_step=6, decay_epoch=1,
                                            step_per_epoch=2)
        optimizer = nn.Adam([{"params": model1_param}, {"params": model2_param}], learning_rate=3e-5)

        mol = MoleculeDataset(data)
        smiles, features = mol.smiles(), mol.features()

        train_loader = ms.dataset.GeneratorDataset(smiles,
                                                   # batch_size=args.batch_size,
                                                   column_names=['smiles'],
                                                   shuffle=True,
                                                   num_parallel_workers=1,
                                                   )

        batch_size = 256
        train_loader = train_loader.batch(batch_size=batch_size)
        # viz = Visdom()
        # viz.line([0.], [0.], win='train_acc', opts=dict(title='train acc'))
        # viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
        df_1 = pd.DataFrame(columns=['time', 'step', 'train Loss'])
        df_1.to_csv("./train_data/train_loss.csv", index=False)


        # Run training
        def forward_fn(batch):
            criterion = ContrastiveLoss(loss_computer='nce_softmax', temperature=args.temperature, args=args)
            emb1 = model1('pretrain', False, batch, None)
            emb2 = model2('pretrain', True, batch, None)
            loss = criterion(emb1, emb2)
            return loss

        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

        # Define function of one-step training
        def train_step(batch):

            loss, grads = grad_fn(batch)
            # print(loss)
            # print(grads)
            optimizer(grads)
            return loss

        def train_loop(dataset, epochs):
            size = dataset.get_dataset_size()
            model1.set_train()
            model2.set_train()
            for batch, (data) in enumerate(dataset.create_tuple_iterator()):
                loss = train_step(data)
                print(loss)
                if batch % 1 == 0:
                    # loss, current = loss.asnumpy(), batch
                    l = loss.asnumpy()
                    print(f"loss: {l:>7f}  [{batch:>3d}/{size:>3d}]")
                with SummaryRecord('./summary_9.5_1') as summary_record:
                    current_step = (epochs - 1) * size + batch
                    # Note2: The output should be a scalar, and use 'add_value' method to record loss.
                    # Note3: You must use the 'record(step)' method to record the data of this step.
                    if current_step % 5 == 0:
                        # summary_record.add_value('scalar', 'loss', loss)
                        # summary_record.record(current_step)
                        # viz.line([loss], [current_step], win='train_acc', update='append')
                        list_loss = [current_step, loss]
                        data_loss = pd.DataFrame([list_loss])
                        data_loss.to_csv("./train_data/train_loss.csv", mode='a', header=False, index=False)

        for epoch in range(args.epochs):
            step_per_schedule = 1
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train_loop(train_loader, epoch + 1)
            # test_loop(test_loader, epoch + 1)
            # test_loop(loader,epoch + 1)
            # if (epoch + 1) % step_per_schedule == 0:
            #     scheduler.step()
            if epoch % 2 == 0:
                # ms.save_checkpoint(model1, "./model_original.ckpt")
                # ms.save_checkpoint(model2, "./model_augment.ckpt")
                ms.save_checkpoint(model1, "./ckpt/model_new1_original"+"%d.ckpt"%(epoch+1))
                ms.save_checkpoint(model2, "./ckpt/model_new1_augment"+"%d.ckpt"%(epoch+1))
        # logger.info(f'[{epoch}/{args.epochs}] train loss {loss.item():.4f}')
    # return embimport numpy as np
from mindspore.common.dtype import List

from .evaluate import evaluate, evaluate_predictions
from chemprop_ms.utils import get_loss_func, get_metric_func, load_checkpoint, makedirs
import os
from argparse import Namespace
from logging import Logger

import mindspore as ms
import mindspore.nn as nn
from mindspore import ParameterTuple
from mindspore import SummaryRecord

from chemprop_ms.models import ContrastiveLoss
from chemprop_ms.torchlight import initialize_exp, snapshot
from chemprop_ms.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop_ms.models import build_model, build_pretrain_model
from chemprop_ms.models.prompt_generator import add_functional_prompt
from chemprop_ms.nn_utils import param_count
from chemprop_ms.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint
from chemprop_ms.data import MoleculeDataset


import pandas as pd



def pre_training(args: Namespace, logger: Logger = None):
    global model2, model1, celllist, model1_param, model2_param, loss
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    '''if args.gpu is not None:
        ms.context.set_context(device_target="GPU", device_id=2)'''

    # Print args
    # =============================================================================
    #     debug(pformat(vars(args)))
    # =============================================================================

    # Get data
    debug('Loading data')
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = 1
    args.data_size = len(data)

    debug(f'Total size = {len(data)}')
    #
    #
    # args.data_size = len(data)
    # debug(f'Total size = {len(data)}')


    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx}')
            # model = build_model(args)
            model1 = build_pretrain_model(args, encoder_name='CMPNN')
            model2 = build_pretrain_model(args, encoder_name='CMPNN')
            celllist = nn.CellList()
            celllist.append(model1)
            celllist.append(model2)
            model1_param = ParameterTuple(celllist[0].trainable_params())
            model2_param = ParameterTuple(celllist[1].trainable_params())

        debug(model1)
        debug(f'Number of M1 parameters = {param_count(model1):,}')
        '''if args.cuda:
            debug('Moving model to cuda')
            model1 = model1.cuda()'''

        debug(model2)
        debug(f'Number of M2 parameters = {param_count(model2):,}')

        logger, dump_folder = initialize_exp(Namespace(**args.__dict__))
        dump_folder = f'{dump_folder}-model'
        '''

        logger, dump_folder = initialize_exp(Namespace(**args.__dict__))
        dump_folder = f'{dump_folder}-model'

        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        args.device = device'''
        ms.set_context(device_target="CPU", device_id=args.gpu, mode=ms.PYNATIVE_MODE)
        # criterion = ContrastiveLoss(loss_computer='nce_softmax', temperature=args.temperature, args=args)
        # optimizer = nn.Adam([{"params": model1.trainable_params()},{"params": model2.trainable_params()}], lr=3e-5)
        # optimizer = Adam([{"params": model1.parameters()}], lr=3e-5)
        # optimizer = nn.Adam(model1.trainable_params(), learning_rate=3e-5)
        scheduler = nn.exponential_decay_lr(learning_rate=3e-4, decay_rate=0.99, total_step=6, decay_epoch=1,
                                            step_per_epoch=2)
        optimizer = nn.Adam([{"params": model1_param}, {"params": model2_param}], learning_rate=3e-5)

        mol = MoleculeDataset(data)
        smiles, features = mol.smiles(), mol.features()

        train_loader = ms.dataset.GeneratorDataset(smiles,
                                                   # batch_size=args.batch_size,
                                                   column_names=['smiles'],
                                                   shuffle=True,
                                                   num_parallel_workers=1,
                                                   )

        batch_size = 256
        train_loader = train_loader.batch(batch_size=batch_size)
        # viz = Visdom()
        # viz.line([0.], [0.], win='train_acc', opts=dict(title='train acc'))
        # viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
        df_1 = pd.DataFrame(columns=['time', 'step', 'train Loss'])
        df_1.to_csv("./train_data/train_loss.csv", index=False)


        # Run training
        def forward_fn(batch):
            criterion = ContrastiveLoss(loss_computer='nce_softmax', temperature=args.temperature, args=args)
            emb1 = model1('pretrain', False, batch, None)
            emb2 = model2('pretrain', True, batch, None)
            loss = criterion(emb1, emb2)
            return loss

        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

        # Define function of one-step training
        def train_step(batch):

            loss, grads = grad_fn(batch)
            # print(loss)
            # print(grads)
            optimizer(grads)
            return loss

        def train_loop(dataset, epochs):
            size = dataset.get_dataset_size()
            model1.set_train()
            model2.set_train()
            for batch, (data) in enumerate(dataset.create_tuple_iterator()):
                loss = train_step(data)
                print(loss)
                if batch % 1 == 0:
                    # loss, current = loss.asnumpy(), batch
                    l = loss.asnumpy()
                    print(f"loss: {l:>7f}  [{batch:>3d}/{size:>3d}]")
                with SummaryRecord('./summary_9.5_1') as summary_record:
                    current_step = (epochs - 1) * size + batch
                    # Note2: The output should be a scalar, and use 'add_value' method to record loss.
                    # Note3: You must use the 'record(step)' method to record the data of this step.
                    if current_step % 5 == 0:
                        # summary_record.add_value('scalar', 'loss', loss)
                        # summary_record.record(current_step)
                        # viz.line([loss], [current_step], win='train_acc', update='append')
                        list_loss = [current_step, loss]
                        data_loss = pd.DataFrame([list_loss])
                        data_loss.to_csv("./train_data/train_loss.csv", mode='a', header=False, index=False)

        for epoch in range(args.epochs):
            step_per_schedule = 1
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train_loop(train_loader, epoch + 1)
            # test_loop(test_loader, epoch + 1)
            # test_loop(loader,epoch + 1)
            # if (epoch + 1) % step_per_schedule == 0:
            #     scheduler.step()
            if epoch % 2 == 0:
                # ms.save_checkpoint(model1, "./model_original.ckpt")
                # ms.save_checkpoint(model2, "./model_augment.ckpt")
                ms.save_checkpoint(model1, "./ckpt/model_new1_original"+"%d.ckpt"%(epoch+1))
                ms.save_checkpoint(model2, "./ckpt/model_new1_augment"+"%d.ckpt"%(epoch+1))
        # logger.info(f'[{epoch}/{args.epochs}] train loss {loss.item():.4f}')
    # return embimport numpy as np
from mindspore.common.dtype import List

from .evaluate import evaluate, evaluate_predictions
from chemprop_ms.utils import get_loss_func, get_metric_func, load_checkpoint, makedirs
import os
from argparse import Namespace
from logging import Logger

import mindspore as ms
import mindspore.nn as nn
from mindspore import ParameterTuple
from mindspore import SummaryRecord

from chemprop_ms.models import ContrastiveLoss
from chemprop_ms.torchlight import initialize_exp, snapshot
from chemprop_ms.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop_ms.models import build_model, build_pretrain_model
from chemprop_ms.models.prompt_generator import add_functional_prompt
from chemprop_ms.nn_utils import param_count
from chemprop_ms.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint
from chemprop_ms.data import MoleculeDataset


import pandas as pd



def pre_training(args: Namespace, logger: Logger = None):
    global model2, model1, celllist, model1_param, model2_param, loss
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    '''if args.gpu is not None:
        ms.context.set_context(device_target="GPU", device_id=2)'''

    # Print args
    # =============================================================================
    #     debug(pformat(vars(args)))
    # =============================================================================

    # Get data
    debug('Loading data')
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = 1
    args.data_size = len(data)

    debug(f'Total size = {len(data)}')
    #
    #
    # args.data_size = len(data)
    # debug(f'Total size = {len(data)}')


    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx}')
            # model = build_model(args)
            model1 = build_pretrain_model(args, encoder_name='CMPNN')
            model2 = build_pretrain_model(args, encoder_name='CMPNN')
            celllist = nn.CellList()
            celllist.append(model1)
            celllist.append(model2)
            model1_param = ParameterTuple(celllist[0].trainable_params())
            model2_param = ParameterTuple(celllist[1].trainable_params())

        debug(model1)
        debug(f'Number of M1 parameters = {param_count(model1):,}')
        '''if args.cuda:
            debug('Moving model to cuda')
            model1 = model1.cuda()'''

        debug(model2)
        debug(f'Number of M2 parameters = {param_count(model2):,}')

        logger, dump_folder = initialize_exp(Namespace(**args.__dict__))
        dump_folder = f'{dump_folder}-model'
        '''

        logger, dump_folder = initialize_exp(Namespace(**args.__dict__))
        dump_folder = f'{dump_folder}-model'

        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        args.device = device'''
        ms.set_context(device_target="CPU", device_id=args.gpu, mode=ms.PYNATIVE_MODE)
        # criterion = ContrastiveLoss(loss_computer='nce_softmax', temperature=args.temperature, args=args)
        # optimizer = nn.Adam([{"params": model1.trainable_params()},{"params": model2.trainable_params()}], lr=3e-5)
        # optimizer = Adam([{"params": model1.parameters()}], lr=3e-5)
        # optimizer = nn.Adam(model1.trainable_params(), learning_rate=3e-5)
        scheduler = nn.exponential_decay_lr(learning_rate=3e-4, decay_rate=0.99, total_step=6, decay_epoch=1,
                                            step_per_epoch=2)
        optimizer = nn.Adam([{"params": model1_param}, {"params": model2_param}], learning_rate=3e-5)

        mol = MoleculeDataset(data)
        smiles, features = mol.smiles(), mol.features()

        train_loader = ms.dataset.GeneratorDataset(smiles,
                                                   # batch_size=args.batch_size,
                                                   column_names=['smiles'],
                                                   shuffle=True,
                                                   num_parallel_workers=1,
                                                   )

        batch_size = 256
        train_loader = train_loader.batch(batch_size=batch_size)
        # viz = Visdom()
        # viz.line([0.], [0.], win='train_acc', opts=dict(title='train acc'))
        # viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
        df_1 = pd.DataFrame(columns=['time', 'step', 'train Loss'])
        df_1.to_csv("./train_data/train_loss.csv", index=False)


        # Run training
        def forward_fn(batch):
            criterion = ContrastiveLoss(loss_computer='nce_softmax', temperature=args.temperature, args=args)
            emb1 = model1('pretrain', False, batch, None)
            emb2 = model2('pretrain', True, batch, None)
            loss = criterion(emb1, emb2)
            return loss

        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

        # Define function of one-step training
        def train_step(batch):

            loss, grads = grad_fn(batch)
            # print(loss)
            # print(grads)
            optimizer(grads)
            return loss

        def train_loop(dataset, epochs):
            size = dataset.get_dataset_size()
            model1.set_train()
            model2.set_train()
            for batch, (data) in enumerate(dataset.create_tuple_iterator()):
                loss = train_step(data)
                print(loss)
                if batch % 1 == 0:
                    # loss, current = loss.asnumpy(), batch
                    l = loss.asnumpy()
                    print(f"loss: {l:>7f}  [{batch:>3d}/{size:>3d}]")
                with SummaryRecord('./summary_9.5_1') as summary_record:
                    current_step = (epochs - 1) * size + batch
                    # Note2: The output should be a scalar, and use 'add_value' method to record loss.
                    # Note3: You must use the 'record(step)' method to record the data of this step.
                    if current_step % 5 == 0:
                        # summary_record.add_value('scalar', 'loss', loss)
                        # summary_record.record(current_step)
                        # viz.line([loss], [current_step], win='train_acc', update='append')
                        list_loss = [current_step, loss]
                        data_loss = pd.DataFrame([list_loss])
                        data_loss.to_csv("./train_data/train_loss.csv", mode='a', header=False, index=False)

        for epoch in range(args.epochs):
            step_per_schedule = 1
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train_loop(train_loader, epoch + 1)
            # test_loop(test_loader, epoch + 1)
            # test_loop(loader,epoch + 1)
            # if (epoch + 1) % step_per_schedule == 0:
            #     scheduler.step()
            if epoch % 2 == 0:
                # ms.save_checkpoint(model1, "./model_original.ckpt")
                # ms.save_checkpoint(model2, "./model_augment.ckpt")
                ms.save_checkpoint(model1, "./ckpt/model_new1_original"+"%d.ckpt"%(epoch+1))
                ms.save_checkpoint(model2, "./ckpt/model_new1_augment"+"%d.ckpt"%(epoch+1))
        # logger.info(f'[{epoch}/{args.epochs}] train loss {loss.item():.4f}')
    # return emb