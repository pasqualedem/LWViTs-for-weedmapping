import time

import gradio as gr
import numpy as np
import pandas as pd
import torch

from wd.app.markdown import grid_summary_builder
from wd.utils.utilities import load_yaml
from wd.utils.segmentation import tensor_to_segmentation_image
from wd.app.inference import WandbInferencer, compose_images

from wd.experiment.experiment import Experimenter

DEFAULT_CMAP = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0)
}

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
        gr.Markdown("Train your model!")
        self.parameters = gr.File(label="Parameter file", value=self.PARAMETERS)
        default_settings = load_yaml(self.PARAMETERS)
        with gr.Row():
            def_mk, def_grids = grid_summary_builder(*self.experimenter.calculate_runs(default_settings))
            self.grids_summary = gr.Markdown(value=def_mk).style()
            self.grids_dataframe = gr.Markdown(value=def_grids)
        self.parameters.change(self.set_parameters_file, inputs=[self.parameters], outputs=[self.grids_summary, self.grids_dataframe])

        self.experiment_btn = gr.Button("Experiment!")
        # self.progress = gr.Label(value={}, label="Progress")
        self.progress = gr.Textbox()
        self.experiment_btn.click(self.experiment, inputs=[], outputs=[self.progress])

    def experiment(self):
        yield self.experimenter.execute_runs(callback=self.update_progress)

    def update_progress(self, cur_grid, cur_run, n_grids, n_runs):
        d = {}
        for i in range(n_grids):
            if i < cur_grid:
                d[f"Grid: {i}"] = 1
            elif i == cur_grid:
                d[f"Grid {i} ({cur_run + 1} / {n_runs})"] = (cur_run + 1) / n_runs
            else:
                d[f"Grid: {i}"] = 0
        yield str(d)

    def set_parameters_file(self, file):
        if file is None:
            return "", ""
        settings = load_yaml(file.name)
        self.experimenter = Experimenter()
        return grid_summary_builder(*self.experimenter.calculate_runs(settings))

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
