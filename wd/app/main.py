import gradio as gr
import numpy as np
import torch

from utils.segmentation import tensor_to_segmentation_image
from wd.app.inference import WandbInferencer, compose_images


def sepia(input_img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img


DEFAULT_CMAP = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0)
}

TITLE = "# Weed Mapping"
TEXT = "Load config and model to select image channels"
DIV = "---"


class Interface:
    def __init__(self):
        self.inferencer = None

        with gr.Blocks() as demo:
            gr.Markdown(TITLE)
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

            demo.launch()

    def predict(self, use_gpu, *inputs):
        if use_gpu:
            self.inferencer.cuda()
        else:
            self.inferencer.cpu()
        inputs = filter(lambda x: x is not None, inputs)
        image = compose_images(list(inputs))
        pred = self.inferencer(image)
        segmentation = self.segment(pred)
        return segmentation

    def segment(self, pred: torch.Tensor):
        pred = pred.squeeze(0).argmax(dim=0).cpu()
        return tensor_to_segmentation_image(pred, DEFAULT_CMAP)

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
