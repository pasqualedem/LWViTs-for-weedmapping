import gc

from super_gradients.training.utils.callbacks import Phase

from wd.callbacks import WandbCallback, SegmentationVisualizationCallback
from wd.data.sequoia import WeedMapDatasetInterface
from wd.experiment.experiment import logger
from wd.experiment.parameters import parse_params
from wd.learning.seg_trainer import SegmentationTrainer
from wd.utils.utilities import dict_to_yaml_string


def run(params: dict):
    seg_trainer = None
    try:
        phases = params['phases']

        train_params, test_params, dataset_params, early_stop = parse_params(params)

        seg_trainer = SegmentationTrainer(experiment_name=params['experiment']['group'],
                                          ckpt_root_dir=params['experiment']['tracking_dir']
                                          if params['experiment']['tracking_dir'] else 'wandb')
        dataset = WeedMapDatasetInterface(dataset_params)
        seg_trainer.connect_dataset_interface(dataset, data_loader_num_workers=params['dataset']['num_workers'])
        seg_trainer.init_model(params, False, None)
        seg_trainer.init_loggers({"in_params": params}, train_params)
        logger.info(f"Input params: \n\n {dict_to_yaml_string(params)}")

        if 'train' in phases:
            train(seg_trainer, train_params, dataset, early_stop)

        if 'test' in phases:
            test_metrics = seg_trainer.test(**test_params)

        if 'inference' in phases:
            inference(seg_trainer, params['run_params'], dataset)
    finally:
        if seg_trainer is not None:
            seg_trainer.sg_logger.close(True)


def train(seg_trainer, train_params, dataset, early_stop):
    # ------------------------ TRAINING PHASE ------------------------
    # Callbacks
    cbcks = [
        WandbCallback(Phase.TRAIN_EPOCH_END, freq=1),
        WandbCallback(Phase.VALIDATION_EPOCH_END, freq=1),
        SegmentationVisualizationCallback(phase=Phase.VALIDATION_BATCH_END,
                                          freq=1,
                                          batch_idxs=[0, len(seg_trainer.train_loader) - 1],
                                          last_img_idx_in_batch=4,
                                          num_classes=dataset.trainset.CLASS_LABELS,
                                          undo_preprocessing=dataset.undo_preprocess),
        *early_stop
    ]
    train_params["phase_callbacks"] = cbcks

    seg_trainer.train(train_params)
    gc.collect()


def inference(seg_trainer, run_params, dataset):
    run_loader = dataset.get_run_loader(folders=run_params['run_folders'], batch_size=run_params['batch_size'])
    cbcks = [
        # SaveSegmentationPredictionsCallback(phase=Phase.POST_TRAINING,
        #                                     path=
        #                                     run_params['prediction_folder']
        #                                     if run_params['prediction_folder'] != 'mlflow'
        #                                     else mlclient.run.info.artifact_uri + '/predictions',
        #                                     num_classes=len(seg_trainer.test_loader.dataset.classes),
        #                                     )
    ]
    run_loader.dataset.return_name = True
    seg_trainer.run(run_loader, callbacks=cbcks)
    # seg_trainer.valid_loader.dataset.return_name = True
    # seg_trainer.run(seg_trainer.valid_loader, callbacks=cbcks)
