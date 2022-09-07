import time

import gradio as gr
import numpy as np
import pandas as pd
import torch
import wandb

from wd.app.markdown import grid_summary_builder
from wd.utils.utilities import load_yaml, update_collection
from wd.utils.segmentation import tensor_to_segmentation_image
from wd.app.inference import WandbInferencer, compose_images

from wd.experiment.experiment import Experimenter

DEFAULT_CMAP = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0)
}

def DEFAULTS(key=None):
    if key is None:
        return "", "", "", "", 0, 0, []
    if key in ['start_from_grid', 'start_from_run']:
        return 0
    else:
        return ""

TITLE = "# Weed Mapping"
TEXT = "Load config and model to select image channels"
DIV = "---"
COLS = ['Grid', 'N. runs']


def segment(pred: torch.Tensor):
    pred = pred.squeeze(0).argmax(dim=0).cpu()
    return tensor_to_segmentation_image(pred, DEFAULT_CMAP)


class Interface:
    PARAMETERS = "./parameters.yaml"

    def __init__(self):
        self.progress = None
        self.experimenter = Experimenter()
        self.grids_dataframe = None
        self.parameters = None
        self.experiment_btn = None
        self.grids_summary = None
        self.inferencer = None

        with gr.Blocks() as demo:
            gr.Markdown(TITLE)
            with gr.Tab("Inference"):
                self.inference_interface()
            with gr.Tab("Training"):
                self.training_inferface()

        demo.queue()
        demo.launch()

    def inference_interface(self):
        use_gpu = gr.components.Checkbox(label="Use GPU")
        with gr.Row():
            model = gr.components.File(type='file', label="Model checkpoint in Wandb folder")
            config = gr.components.File(type='file', label="Wandb config input")
        text = gr.Textbox(TEXT, show_label=False)
        divider = gr.Markdown(DIV)
        with gr.Row() as self.input_images_row:
            self.input_channels = {
                'R': gr.Image(type='pil', image_mode='L', label='R Channel', visible=False),
                'G': gr.Image(type='pil', image_mode='L', label='G Channel', visible=False),
                'B': gr.Image(type='pil', image_mode='L', label='B Channel', visible=False),
                "NDVI": gr.Image(type='pil', image_mode='L', label='NDVI Channel', visible=False),
                "NIR": gr.Image(type='pil', image_mode='L', label='NIR Channel', visible=False),
            }
        config.change(self.set_model,
                      inputs=[model, config, use_gpu],
                      outputs=[*self.input_channels.values(), text],
                      )
        model.change(self.set_model,
                     inputs=[model, config, use_gpu],
                     outputs=[*self.input_channels.values(), text],
                     )
        submit = gr.Button(variant="primary")
        segmentation = gr.Image(label="Segmentation")

        submit.click(self.predict, inputs=[use_gpu, *self.input_channels.values()], outputs=[segmentation])

    def training_inferface(self):
        default_settings = load_yaml(self.PARAMETERS)
        def_exp_settings = default_settings['experiment']
        gr.Markdown("Train your model!")
        with gr.Row():
            self.group_txt = gr.Textbox(label="Experiment group", value=def_exp_settings.get('group') or '')
            self.track_dir = gr.Textbox(label="Tracking directory", value=def_exp_settings.get('tracking_dir') or '')
            self.start_from_grid_txt = gr.Number(label="Start from grid", value=def_exp_settings.get('start_from_grid') or 0)
            self.start_from_run_txt = gr.Number(label="Start from run", value=def_exp_settings.get('start_from_run') or 0)
            flags = []
            if def_exp_settings.get('resume'):
                flags.append('resume')
            if def_exp_settings.get('continue_with_errors'):
                flags.append('continue_with_errors')
            self.flags = gr.CheckboxGroup(["resume", "continue_with_errors"], label="", value=flags)

        self.parameters = gr.File(label="Parameter file", value=self.PARAMETERS)
        with gr.Row():
            def_mk, def_grids = grid_summary_builder(*self.experimenter.calculate_runs(default_settings))
            self.grids_summary = gr.Markdown(value=def_mk).style()
            self.grids_dataframe = gr.Markdown(value=def_grids)

        parameters_comps = [self.group_txt,
                            self.track_dir, self.start_from_grid_txt,
                            self.start_from_run_txt, self.flags]
        self.parameters.change(self.set_parameters_file, inputs=[self.parameters],
                               outputs=[self.grids_summary, self.grids_dataframe] + parameters_comps)

        with gr.Row():
            self.stop_btn = gr.Button("Stop!")
            self.experiment_btn = gr.Button("Experiment!", variant="primary")
        self.progress = gr.Label(value={}, label="Progress")
        self.experiment_btn.click(self.experiment, inputs=[], outputs=[self.progress])
        self.txt = gr.Textbox("no")
        self.stop_btn.click(self.stop, inputs=[], outputs=[self.txt])

    def experiment(self):
        yield self.update_progress(self.experimenter.gs.starting_grid,
                                   self.experimenter.gs.starting_run - 1,
                                   len(self.experimenter.grids),
                                   len(self.experimenter.grids[self.experimenter.gs.starting_grid]),
                                   True)
        for out in self.experimenter.execute_runs(callback=self.update_progress):
            yield out

    def stop(self):
        return "Parallel"

    def update_progress(self, cur_grid, cur_run, n_grids, n_runs, success):
        d = {}
        for i in range(n_grids):
            if i < cur_grid:
                d[f"Grid {i}"] = 1
            elif i == cur_grid:
                d[f"Grid {i} ({cur_run + 1} / {n_runs})"] = (cur_run + 1) / n_runs
            else:
                d[f"Grid {i}"] = 0
        return d

    def set_parameters_file(self, file):
        if file is None:
            self.experimenter.grids = None
            return DEFAULTS()
        settings = load_yaml(file.name)
        if self.experimenter is not None:
            if self.experimenter.exp_settings is not None:
                d = {
                    k: self.experimenter.exp_settings.get(k) or DEFAULTS(k) for k in
                    ['resume', 'tracking_dir', 'group', 'start_from_grid', 'start_from_run', 'continue_with_errors']
                }
                settings['experiment'] = update_collection(settings['experiment'], d)

        self.experimenter = Experimenter()
        sum_mk, params_mk = grid_summary_builder(*self.experimenter.calculate_runs(settings))
        flags = []
        if self.experimenter.exp_settings.get('resume'):
            flags.append('resume')
        if self.experimenter.exp_settings.get('continue_with_errors'):
            flags.append('continue_with_errors')
        return sum_mk, params_mk, \
               self.experimenter.exp_settings['group'], \
               self.experimenter.exp_settings['tracking_dir'] or DEFAULTS('tracking_dir'), \
               self.experimenter.exp_settings['start_from_grid'], \
               self.experimenter.exp_settings['start_from_run'], \
               flags

    def set_exp_settings(self, name, track_dir, start_grid, start_run, flags):
        if self.experimenter is None:
            self.experimenter = Experimenter()
        if self.experimenter.exp_settings is None:
            self.experimenter.exp_settings = {}
        self.experimenter.exp_settings = update_collection(self.experimenter.exp_settings,
                                                           {
                                                               "resume": ("resume" in flags),
                                                               "continue_with_errors": (
                                                                       "continue_with_errors" in flags),
                                                               "start_from_grid": start_grid,
                                                               "start_from_run": start_run,
                                                               "tracking_dir": track_dir,
                                                               "group": name,
                                                           })

    def predict(self, use_gpu, *inputs):
        if use_gpu:
            self.inferencer.cuda()
        else:
            self.inferencer.cpu()
        inputs = filter(lambda x: x is not None, inputs)
        image = compose_images(list(inputs))
        pred = self.inferencer(image)
        segmentation = segment(pred)
        return segmentation

    def set_model(self, model_path_wrapper, config_wrapper, gpu):
        if model_path_wrapper is None or config_wrapper is None:
            return [gr.update(visible=False) for _ in self.input_channels] + [TEXT]
        self.inferencer = WandbInferencer(model_path_wrapper, config_wrapper, gpu)
        input_channels = []
        for channel, input_form in self.input_channels.items():
            if channel in self.inferencer.channels:
                input_channels.append(gr.update(visible=True))
            else:
                input_channels.append(gr.update(visible=False))
        return input_channels + ["".join([f"{channel} " for channel in self.inferencer.channels]) + "needed"]


def frontend():
    Interface()
