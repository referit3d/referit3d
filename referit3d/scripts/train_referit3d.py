#!/usr/bin/env python
# coding: utf-8

import torch
import tqdm
import time
import warnings
import os.path as osp
import torch.nn as nn
from torch import optim
from termcolor import colored

from referit3d.in_out.arguments import parse_arguments
from referit3d.in_out.neural_net_oriented import load_scan_related_data, load_referential_data
from referit3d.in_out.neural_net_oriented import compute_auxiliary_data, trim_scans_per_referit3d_data
from referit3d.in_out.pt_datasets.listening_dataset import make_data_loaders
from referit3d.utils import set_gpu_to_zero_position, create_logger, seed_training_code
from referit3d.utils.tf_visualizer import Visualizer
from referit3d.models.referit3d_net import instantiate_referit3d_net
from referit3d.models.referit3d_net_utils import single_epoch_train, evaluate_on_dataset
from referit3d.models.utils import load_state_dicts, save_state_dicts
from referit3d.analysis.deepnet_predictions import analyze_predictions

if __name__ == '__main__':
    def log_train_test_information():
        """Helper logging function.
        Note uses "global" variables defined below.
        """
        logger.info('Epoch:{}'.format(epoch))
        for phase in ['train', 'test']:
            if phase == 'train':
                meters = train_meters
            else:
                meters = test_meters

            info = '{}: Total-Loss {:.4f}, Listening-Acc {:.4f}'.format(phase,
                                                                        meters[phase + '_total_loss'],
                                                                        meters[phase + '_referential_acc'])

            if args.obj_cls_alpha > 0:
                info += ', Object-Clf-Acc: {:.4f}'.format(meters[phase + '_object_cls_acc'])

            if args.lang_cls_alpha > 0:
                info += ', Text-Clf-Acc: {:.4f}'.format(meters[phase + '_txt_cls_acc'])

            logger.info(info)
            logger.info('{}: Epoch-time {:.3f}'.format(phase, timings[phase]))
        logger.info('Best so far {:.3f} (@epoch {})'.format(best_test_acc, best_test_epoch))


    # Parse arguments
    args = parse_arguments()

    # Read the scan related information
    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(args.scannet_file)

    # Read the linguistic data of ReferIt3D
    referit_data = load_referential_data(args, args.referit3D_file, scans_split)

    # Prepare data & compute auxiliary meta-information.
    all_scans_in_dict = trim_scans_per_referit3d_data(referit_data, all_scans_in_dict)
    mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans_in_dict, args)
    data_loaders = make_data_loaders(args, referit_data, vocab, class_to_idx, all_scans_in_dict, mean_rgb)

    # Prepare GPU environment
    set_gpu_to_zero_position(args.gpu)  # Pnet++ seems to work only at "gpu:0"
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    seed_training_code(args.random_seed)

    # Losses:
    criteria = dict()

    # Referential, "find the object in the scan" loss
    if args.s_vs_n_weight is not None:  # TODO - move to a better place
        assert args.augment_with_sr3d is not None
        ce = nn.CrossEntropyLoss(reduction='none').to(device)
        s_vs_n_weight = args.s_vs_n_weight


        def weighted_ce(logits, batch):
            loss_per_example = ce(logits, batch['target_pos'])
            sr3d_mask = ~batch['is_nr3d']
            loss_per_example[sr3d_mask] *= s_vs_n_weight
            loss = loss_per_example.sum() / len(loss_per_example)
            return loss


        criteria['logits'] = weighted_ce
    else:
        criteria['logits'] = nn.CrossEntropyLoss().to(device)

    # Object-type classification
    if args.obj_cls_alpha > 0:
        criteria['class_logits'] = nn.CrossEntropyLoss(ignore_index=class_to_idx['pad']).to(device)

    # Target-in-language guessing
    if args.lang_cls_alpha > 0:
        criteria['lang_logits'] = nn.CrossEntropyLoss().to(device)

    # Prepare the Listener
    n_classes = len(class_to_idx) - 1  # -1 to ignore the <pad> class
    pad_idx = class_to_idx['pad']

    model = instantiate_referit3d_net(args, vocab, n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.65,
                                                              patience=5, verbose=True)

    start_training_epoch = 1
    best_test_acc = -1
    best_test_epoch = -1
    no_improvement = 0

    if args.resume_path:
        warnings.warn('Resuming assumes that the BEST per-val model is loaded!')
        # perhaps best_test_acc, best_test_epoch, best_test_epoch =  unpickle...
        loaded_epoch = load_state_dicts(args.resume_path, map_location=device, model=model)
        print('Loaded a model stopped at epoch: {}.'.format(loaded_epoch))
        if not args.fine_tune:
            print('Loaded a model that we do NOT plan to fine-tune.')
            load_state_dicts(args.resume_path, optimizer=optimizer, lr_scheduler=lr_scheduler)
            start_training_epoch = loaded_epoch + 1
            best_test_epoch = loaded_epoch
            best_test_acc = lr_scheduler.best
            print('Loaded model had {} test-accuracy in the corresponding dataset used when trained.'.format(
                best_test_acc))
        else:
            print('Parameters that do not allow gradients to be back-propped:')
            ft_everything = True
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    print(name)
                    exist = False
            if ft_everything:
                print('None, all wil be fine-tuned')
            # if you fine-tune the previous epochs/accuracy are irrelevant.
            dummy = args.max_train_epochs + 1 - start_training_epoch
            print('Ready to *fine-tune* the model for a max of {} epochs'.format(dummy))

    # Training.
    if args.mode == 'train':
        train_vis = Visualizer(args.tensorboard_dir)
        logger = create_logger(args.log_dir)
        logger.info('Starting the training. Good luck!')

        with tqdm.trange(start_training_epoch, args.max_train_epochs + 1, desc='epochs') as bar:
            timings = dict()
            for epoch in bar:
                # Train:
                tic = time.time()
                train_meters = single_epoch_train(model, data_loaders['train'], criteria, optimizer,
                                                  device, pad_idx, args=args)
                toc = time.time()
                timings['train'] = (toc - tic) / 60

                # Evaluate:
                tic = time.time()
                test_meters = evaluate_on_dataset(model, data_loaders['test'], criteria, device, pad_idx, args=args)
                toc = time.time()
                timings['test'] = (toc - tic) / 60

                eval_acc = test_meters['test_referential_acc']
                lr_scheduler.step(eval_acc)

                if best_test_acc < eval_acc:
                    logger.info(colored('Test accuracy, improved @epoch {}'.format(epoch), 'green'))
                    best_test_acc = eval_acc
                    best_test_epoch = epoch

                    # Save the model (overwrite the best one)
                    save_state_dicts(osp.join(args.checkpoint_dir, 'best_model.pth'),
                                     epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
                    no_improvement = 0
                else:
                    no_improvement += 1
                    logger.info(colored('Test accuracy, did not improve @epoch {}'.format(epoch), 'red'))

                log_train_test_information()
                train_meters.update(test_meters)
                train_vis.log_scalars({k: v for k, v in train_meters.items() if '_acc' in k}, step=epoch,
                                      main_tag='acc')
                train_vis.log_scalars({k: v for k, v in train_meters.items() if '_loss' in k},
                                      step=epoch, main_tag='loss')

                bar.refresh()

                if no_improvement == args.patience:
                    logger.warning(colored('Stopping the training @epoch-{} due to lack of progress in test-accuracy '
                                           'boost (patience hit {} epochs)'.format(epoch, args.patience),
                                           'red', attrs=['bold', 'underline']))
                    break

        with open(osp.join(args.checkpoint_dir, 'final_result.txt'), 'w') as f_out:
            msg = ('Best accuracy: {:.4f} (@epoch {})'.format(best_test_acc, best_test_epoch))
            f_out.write(msg)

        logger.info('Finished training successfully. Good job!')

    elif args.mode == 'evaluate':

        meters = evaluate_on_dataset(model, data_loaders['test'], criteria, device, pad_idx, args=args)
        print('Reference-Accuracy: {:.4f}'.format(meters['test_referential_acc']))
        print('Object-Clf-Accuracy: {:.4f}'.format(meters['test_object_cls_acc']))
        print('Text-Clf-Accuracy {:.4f}:'.format(meters['test_txt_cls_acc']))

        out_file = osp.join(args.checkpoint_dir, 'test_result.txt')
        res = analyze_predictions(model, data_loaders['test'].dataset, class_to_idx, pad_idx, device,
                                  args, out_file=out_file)
        print(res)
