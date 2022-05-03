import json
import os
import signal
import time
from typing import Optional, Union, Any

import psutil
import torch
import wandb
from PIL import Image
from flatbuffers.builder import np
from matplotlib import pyplot as plt
from super_gradients.common import ADNNModelRepositoryDataInterfaces
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.environment.env_helpers import multi_process_safe
from super_gradients.common.sg_loggers.abstract_sg_logger import AbstractSGLogger
from super_gradients.training.params import TrainingParams
from super_gradients.training.utils import sg_model_utils


logger = get_logger(__name__)

WANDB_ID_PREFIX = 'wandb_id.'
WANDB_INCLUDE_FILE_NAME = '.wandbinclude'


class BaseSGLogger(AbstractSGLogger):

    def __init__(self, project_name: str,
                 experiment_name: str,
                 storage_location: str,
                 resumed: bool,
                 training_params: TrainingParams,
                 checkpoints_dir_path: str,
                 tb_files_user_prompt: bool = False,
                 launch_tensorboard: bool = False,
                 tensorboard_port: int = None,
                 save_checkpoints_remote: bool = True,
                 save_tensorboard_remote: bool = True,
                 save_logs_remote: bool = True):
        """

        :param experiment_name: Used for logging and loading purposes
        :param storage_location: If set to 's3' (i.e. s3://my-bucket) saves the Checkpoints in AWS S3 otherwise saves the Checkpoints Locally
        :param resumed: if true, then old tensorboard files will *not* be deleted when tb_files_user_prompt=True
        :param training_params: training_params for the experiment.
        :param checkpoints_dir_path: Local root directory path where all experiment logging directories will
                                                 reside.
        :param tb_files_user_prompt: Asks user for Tensorboard deletion prompt.
        :param launch_tensorboard: Whether to launch a TensorBoard process.
        :param tensorboard_port: Specific port number for the tensorboard to use when launched (when set to None, some free port
                    number will be used
        :param save_checkpoints_remote: Saves checkpoints in s3.
        :param save_tensorboard_remote: Saves tensorboard in s3.
        :param save_logs_remote: Saves log files in s3.
        """
        super().__init__()
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.storage_location = storage_location

        if storage_location.startswith('s3'):
            self.save_checkpoints_remote = save_checkpoints_remote
            self.save_tensorboard_remote = save_tensorboard_remote
            self.save_logs_remote = save_logs_remote
            self.remote_storage_available = True
        else:
            self.remote_storage_available = False
            if save_checkpoints_remote:
                logger.error('save_checkpoints_remote == True but storage_location is not s3 path. Files will not be saved remotely')
            if save_tensorboard_remote:
                logger.error('save_tensorboard_remote == True but storage_location is not s3 path. Files will not be saved remotely')
            if save_logs_remote:
                logger.error('save_logs_remote == True but storage_location is not s3 path. Files will not be saved remotely')

            self.save_checkpoints_remote = False
            self.save_tensorboard_remote = False
            self.save_logs_remote = False

        self.tensor_board_process = None
        self.max_global_steps = training_params.max_epochs
        self._local_dir = checkpoints_dir_path

        self._make_dir()
        self._init_log_file()

        self.model_checkpoints_data_interface = ADNNModelRepositoryDataInterfaces(data_connection_location=self.storage_location)

        self.use_tensorboard = launch_tensorboard
        if launch_tensorboard:
            self._launch_tensorboard(port=tensorboard_port)
            self._init_tensorboard(resumed, tb_files_user_prompt)

    @multi_process_safe
    def _launch_tensorboard(self, port):
        self.tensor_board_process, _ = sg_model_utils.launch_tensorboard_process(self._local_dir, port=port)

    @multi_process_safe
    def _init_tensorboard(self, resumed, tb_files_user_prompt):
        self.tensorboard_writer = sg_model_utils.init_summary_writer(self._local_dir, resumed, tb_files_user_prompt)

    @multi_process_safe
    def _make_dir(self):
        if not os.path.isdir(self._local_dir):
            os.makedirs(self._local_dir)

    @multi_process_safe
    def _init_log_file(self):
        time_string = time.strftime('%b%d_%H_%M_%S', time.localtime())
        self.log_file_path = f'{self._local_dir}/log_{time_string}.txt'

    @multi_process_safe
    def _write_to_log_file(self, lines: list):
        with open(self.log_file_path, 'a' if os.path.exists(self.log_file_path) else 'w') as log_file:
            for line in lines:
                log_file.write(line + '\n')

    @multi_process_safe
    def add_config(self, tag: str, config: dict):
        log_lines = ['--------- config parameters ----------']
        log_lines.append(json.dumps(config, indent=4, default=str))
        log_lines.append('------- config parameters end --------')

        if self.use_tensorboard:
            self.tensorboard_writer.add_text("Hyper_parameters", json.dumps(config, indent=4, default=str)
                                             .replace(" ", "&nbsp;").replace("\n", "  \n  "))
        self._write_to_log_file(log_lines)

    @multi_process_safe
    def add_scalar(self, tag: str, scalar_value: float, global_step: int = None):
        if self.use_tensorboard:
            self.tensorboard_writer.add_scalar(tag=tag.lower().replace(' ', '_'), scalar_value=scalar_value, global_step=global_step)

    @multi_process_safe
    def add_scalars(self, tag_scalar_dict: dict, global_step: int = None):
        """
        add multiple scalars.
        Unlike Tensorboard implementation, this does not add all scalars with a main tag (all scalars to the same chart).
        Instead, scalars are added to tensorboard like in add_scalar and are written in log together.
        """
        if self.use_tensorboard:
            for tag, value in tag_scalar_dict.items():
                self.tensorboard_writer.add_scalar(tag=tag.lower().replace(' ', '_'), scalar_value=value, global_step=global_step)

            self.tensorboard_writer.flush()

        # WRITE THE EPOCH RESULTS TO LOG FILE
        log_line = f'\nEpoch ({global_step}/{self.max_global_steps})  - '
        for tag, value in tag_scalar_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            log_line += f'{tag.replace(" ", "_")}: {value}\t'

        self._write_to_log_file([log_line])

    @multi_process_safe
    def add_image(self, tag: str, image: Union[torch.Tensor, np.array, Image.Image], data_format='CHW', global_step: int = None):
        if self.use_tensorboard:
            self.tensorboard_writer.add_image(tag=tag, img_tensor=image, dataformats=data_format, global_step=global_step)

    @multi_process_safe
    def add_images(self, tag: str, images: Union[torch.Tensor, np.array], data_format='NCHW', global_step: int = None):
        """
        Add multiple images to SGLogger.
        Typically, this function will add a set of images to tensorboard, save them to disk or add it to experiment management framework.

        :param tag: Data identifier
        :param images: images to be added. The values should lie in [0, 255] for type uint8 or [0, 1] for type float.
        :param data_format: Image data format specification of the form NCHW, NHWC, CHW, HWC, HW, WH, etc.
        :param global_step: Global step value to record
        """
        if self.use_tensorboard:
            self.tensorboard_writer.add_images(tag=tag, img_tensor=images, dataformats=data_format, global_step=global_step)

    @multi_process_safe
    def add_video(self, tag: str, video: Union[torch.Tensor, np.array], global_step: int = None):
        """
        Add a single video to SGLogger.
        Typically, this function will add a video to tensorboard, save it to disk or add it to experiment management framework.

        :param tag: Data identifier
        :param video: the video to add. shape (N,T,C,H,W) or (T,C,H,W). The values should lie in [0, 255] for type uint8 or [0, 1] for type float.
        :param global_step: Global step value to record
        """
        if video.ndim < 5:
            video = video[None, ]
        if self.use_tensorboard:
            self.tensorboard_writer.add_video(tag=tag, video=video, global_step=global_step)

    @multi_process_safe
    def add_histogram(self, tag: str, values: Union[torch.Tensor, np.array], bins: str, global_step: int = None):
        if self.use_tensorboard:
            self.tensorboard_writer.add_histogram(tag=tag, values=values, bins=bins, global_step=global_step)

    @multi_process_safe
    def add_model_graph(self, tag: str, model: torch.nn.Module, dummy_input: torch.Tensor):
        """
        Add a pytorch model graph to the SGLogger.
        Only the model structure/architecture will be preserved and collected, NOT the model weights.

        :param tag: Data identifier
        :param model: the model to be added
        :param dummy_input: an input to be used for a forward call on the model
        """
        if self.use_tensorboard:
            self.tensorboard_writer.add_graph(model=model, input_to_model=dummy_input)

    @multi_process_safe
    def add_text(self, tag: str, text_string: str, global_step: int = None):
        if self.use_tensorboard:
            self.tensorboard_writer.add_text(tag=tag, text_string=text_string, global_step=global_step)

    @multi_process_safe
    def add_figure(self, tag: str, figure: plt.figure, global_step: int = None):
        """
        Add a text to SGLogger.
        Typically, this function will add a figure to tensorboard or add it to experiment management framework.

        :param tag: Data identifier
        :param figure: the figure to add
        :param global_step: Global step value to record
        """
        if self.use_tensorboard:
            self.tensorboard_writer.add_figure(tag=tag, figure=figure, global_step=global_step)

    @multi_process_safe
    def add_file(self, file_name: str = None):
        if self.remote_storage_available and self.use_tensorboard:
            self.model_checkpoints_data_interface.save_remote_tensorboard_event_files(self.experiment_name, self._local_dir, file_name)

    @multi_process_safe
    def upload(self):
        if self.save_tensorboard_remote and self.use_tensorboard:
            self.model_checkpoints_data_interface.save_remote_tensorboard_event_files(self.experiment_name, self._local_dir)

        if self.save_logs_remote:
            log_file_name = self.log_file_path.split('/')[-1]
            self.model_checkpoints_data_interface.save_remote_checkpoints_file(self.experiment_name, self._local_dir, log_file_name)

    @multi_process_safe
    def flush(self):
        if self.use_tensorboard:
            self.tensorboard_writer.flush()

    @multi_process_safe
    def close(self):
        if self.use_tensorboard:
            self.tensorboard_writer.close()
        if self.tensor_board_process is not None:
            try:
                logger.info('[CLEANUP] - Stopping tensorboard process')
                process = psutil.Process(self.tensor_board_process.pid)
                process.send_signal(signal.SIGTERM)
                logger.info('[CLEANUP] - Successfully stopped tensorboard process')
            except Exception as ex:
                logger.info('[CLEANUP] - Could not stop tensorboard process properly: ' + str(ex))

    @multi_process_safe
    def add_checkpoint(self, tag: str, state_dict: dict, global_step: int = None):

        name = f'ckpt_{global_step}.pth' if tag is None else tag
        if not name.endswith('.pth'):
            name += '.pth'

        path = os.path.join(self._local_dir, name)
        torch.save(state_dict, path)
        if tag == 'ckpt_best.pth':
            logger.info("Checkpoint saved in " + path)
        if self.save_checkpoints_remote:
            self.model_checkpoints_data_interface.save_remote_checkpoints_file(self.experiment_name, self._local_dir, name)

    def add(self, tag: str, obj: Any, global_step: int = None):
        pass

    def local_dir(self) -> str:
        return self._local_dir


class WandBSGLogger(BaseSGLogger):

    def __init__(self, project_name: str, experiment_name: str, storage_location: str, resumed: bool,
                 training_params: dict, checkpoints_dir_path: str, tb_files_user_prompt: bool = False,
                 launch_tensorboard: bool = False, tensorboard_port: int = None, save_checkpoints_remote: bool = True,
                 save_tensorboard_remote: bool = True, save_logs_remote: bool = True, entity: Optional[str] = None,
                 api_server: Optional[str] = None, save_code: bool = False, tags=None, run_id=None, **kwargs):
        """

        :param experiment_name: Used for logging and loading purposes
        :param s3_path: If set to 's3' (i.e. s3://my-bucket) saves the Checkpoints in AWS S3 otherwise saves the Checkpoints Locally
        :param checkpoint_loaded: if true, then old tensorboard files will *not* be deleted when tb_files_user_prompt=True
        :param max_epochs: the number of epochs planned for this training
        :param tb_files_user_prompt: Asks user for Tensorboard deletion prompt.
        :param launch_tensorboard: Whether to launch a TensorBoard process.
        :param tensorboard_port: Specific port number for the tensorboard to use when launched (when set to None, some free port
                    number will be used
        :param save_checkpoints_remote: Saves checkpoints in s3.
        :param save_tensorboard_remote: Saves tensorboard in s3.
        :param save_logs_remote: Saves log files in s3.
        :param save_code: save current code to wandb
        """
        self.s3_location_available = storage_location.startswith('s3')
        self.resumed = resumed
        resume = 'must' if resumed else None
        os.makedirs(checkpoints_dir_path, exist_ok=True)
        run = wandb.init(project=project_name, name=experiment_name,
                         entity=entity, resume=resume, id=run_id, tags=tags,
                         dir=checkpoints_dir_path, **kwargs)
        if save_code:
            self._save_code()

        self.save_checkpoints_wandb = save_checkpoints_remote
        self.save_tensorboard_wandb = save_tensorboard_remote
        self.save_logs_wandb = save_logs_remote
        checkpoints_dir_path = os.path.relpath(run.dir, os.getcwd())
        super().__init__(project_name, experiment_name, storage_location, resumed, training_params,
                         checkpoints_dir_path, tb_files_user_prompt, launch_tensorboard, tensorboard_port,
                         self.s3_location_available, self.s3_location_available, self.s3_location_available)

        self._set_wandb_id(run.id)
        if api_server is not None:
            if api_server != os.getenv('WANDB_BASE_URL'):
                logger.warning(f'WANDB_BASE_URL environment parameter not set to {api_server}. Setting the parameter')
                os.putenv('WANDB_BASE_URL', api_server)

    @multi_process_safe
    def _save_code(self):
        """
        Save the current code to wandb.
        If a file named .wandbinclude is avilable in the root dir of the project the settings will be taken from the file.
        Otherwise, all python file in the current working dir (recursively) will be saved.
        File structure: a single relative path or a single type in each line.
        i.e:

        src
        tests
        examples
        *.py
        *.yaml

        The paths and types in the file are the paths and types to be included in code upload to wandb
        """
        base_path, paths, types = self._get_include_paths()

        if len(types) > 0:
            def func(path):
                for p in paths:
                    if path.startswith(p):
                        for t in types:
                            if path.endswith(t):
                                return True
                return False

            include_fn = func
        else:
            include_fn = lambda path: path.endswith(".py")

        if base_path != ".":
            wandb.run.log_code(base_path, include_fn=include_fn)
        else:
            wandb.run.log_code(".", include_fn=include_fn)


    @multi_process_safe
    def add_config(self, tag: str = None, config: dict = None):
        if tag:
            config = {tag: config}
        wandb.config.update(config, allow_val_change=self.resumed)

    @multi_process_safe
    def add_scalar(self, tag: str, scalar_value: float, global_step: int = 0):
        wandb.log(data={tag: scalar_value}, step=global_step)

    @multi_process_safe
    def add_scalars(self, tag_scalar_dict: dict, global_step: int = 0):
        wandb.log(data=tag_scalar_dict, step=global_step)

    @multi_process_safe
    def add_image(self, tag: str, image: Union[torch.Tensor, np.array, Image.Image], data_format='CHW', global_step: int = 0):
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
        if image.shape[0] < 5:
            image = image.transpose([1, 2, 0])
        wandb.log(data={tag: wandb.Image(image, caption=tag)}, step=global_step)

    @multi_process_safe
    def add_images(self, tag: str, images: Union[torch.Tensor, np.array], data_format='NCHW', global_step: int = 0):

        wandb_images = []
        for im in images:
            if isinstance(im, torch.Tensor):
                im = im.cpu().detach().numpy()

            if im.shape[0] < 5:
                im = im.transpose([1, 2, 0])
            wandb_images.append(wandb.Image(im))
        wandb.log({tag: wandb_images}, step=global_step)

    @multi_process_safe
    def add_video(self, tag: str, video: Union[torch.Tensor, np.array], global_step: int = 0):

        if video.ndim > 4:
            for index, vid in enumerate(video):
                self.add_video(tag=f'{tag}_{index}', video=vid, global_step=global_step)
        else:
            if isinstance(video, torch.Tensor):
                video = video.cpu().detach().numpy()
            wandb.log({tag: wandb.Video(video, fps=4)}, step=global_step)

    @multi_process_safe
    def add_histogram(self, tag: str, values: Union[torch.Tensor, np.array], bins: str, global_step: int = 0):
        wandb.log({tag: wandb.Histogram(values, num_bins=bins)}, step=global_step)

    @multi_process_safe
    def add_text(self, tag: str, text_string: str, global_step: int = 0):
        wandb.log({tag: text_string}, step=global_step)

    @multi_process_safe
    def add_figure(self, tag: str, figure: plt.figure, global_step: int = 0):
        wandb.log({tag: figure}, step=global_step)

    @multi_process_safe
    def add_table(self, tag, data, columns, rows):
        if isinstance(data, torch.Tensor):
            data = [[x.item() for x in row] for row in data]
        table = wandb.Table(data=data, rows=rows, columns=columns)
        wandb.log({tag: table})

    @multi_process_safe
    def close(self, really=False):
        if really:
            super().close()
            wandb.finish()

    @multi_process_safe
    def add_file(self, file_name: str = None):
        wandb.save(glob_str=os.path.join(self._local_dir, file_name), base_path=self._local_dir, policy='now')

    @multi_process_safe
    def add_summary(self, metrics: dict):
        wandb.summary.update(metrics)

    @multi_process_safe
    def upload(self):

        if self.save_tensorboard_wandb:
            wandb.save(glob_str=self._get_tensorboard_file_name(), base_path=self._local_dir, policy='now')

        if self.save_logs_wandb:
            wandb.save(glob_str=self.log_file_path, base_path=self._local_dir, policy='now')

    @multi_process_safe
    def add_checkpoint(self, tag: str, state_dict: dict, global_step: int = 0):
        name = f'ckpt_{global_step}.pth' if tag is None else tag
        if not name.endswith('.pth'):
            name += '.pth'

        path = os.path.join(self._local_dir, name)
        torch.save(state_dict, path)

        if self.save_checkpoints_wandb:
            if self.s3_location_available:
                self.model_checkpoints_data_interface.save_remote_checkpoints_file(self.experiment_name, self._local_dir, name)
            wandb.save(glob_str=path, base_path=self._local_dir, policy='now')

    def _get_tensorboard_file_name(self):
        try:
            tb_file_path = self.tensorboard_writer.file_writer.event_writer._file_name
        except RuntimeError as e:
            logger.warning('tensorboard file could not be located for ')
            return None

        return tb_file_path

    def _get_wandb_id(self):
        for file in os.listdir(self._local_dir):
            if file.startswith(WANDB_ID_PREFIX):
                return file.replace(WANDB_ID_PREFIX, '')

    def _set_wandb_id(self, id):
        for file in os.listdir(self._local_dir):
            if file.startswith(WANDB_ID_PREFIX):
                os.remove(os.path.join(self._local_dir, file))

    def add(self, tag: str, obj: Any, global_step: int = None):
        pass

    def _get_include_paths(self):
        """
        Look for .wandbinclude file in parent dirs and return the list of paths defined in the file.

        file structure is a single relative (i.e. src/) or a single type (i.e *.py)in each line.
        the paths and types in the file are the paths and types to be included in code upload to wandb
        :return: if file exists, return the list of paths and a list of types defined in the file
        """

        wandb_include_file_path = self._search_upwards_for_file(WANDB_INCLUDE_FILE_NAME)
        if wandb_include_file_path is not None:
            with open(wandb_include_file_path) as file:
                lines = file.readlines()

            base_path = os.path.dirname(wandb_include_file_path)
            paths = []
            types = []
            for line in lines:
                line = line.strip().strip('/n')
                if line == "" or line.startswith("#"):
                    continue

                if line.startswith('*.'):
                    types.append(line.replace('*', ''))
                else:
                    paths.append(os.path.join(base_path, line))
            return base_path, paths, types

        return ".", [], []

    @staticmethod
    def _search_upwards_for_file(file_name: str):
        """
        Search in the current directory and all directories above it for a file of a particular name.
        :param file_name: file name to look for.
        :return: pathlib.Path, the location of the first file found or None, if none was found
        """

        try:
            cur_dir = os.getcwd()
            while cur_dir != '/':
                if file_name in os.listdir(cur_dir):
                    return os.path.join(cur_dir, file_name)
                else:
                    cur_dir = os.path.dirname(cur_dir)
        except RuntimeError as e:
            return None

        return None

    def __repr__(self):
        return "WandbSGLogger"
