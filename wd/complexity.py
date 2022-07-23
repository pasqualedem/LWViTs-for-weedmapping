import numpy as np
import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
from models import MODELS


def seg_model_flops(model, size, verbose=False, per_layer_stats=False, model_args={}):
    n_channels, w, h = size
    net = MODELS[model]({'input_channels': n_channels, 'num_classes': 3, "output_channels": 3, **model_args})
    macs, params = get_model_complexity_info(net, (n_channels, w, h), as_strings=True,
                                             print_per_layer_stat=per_layer_stats, verbose=verbose)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def seg_inference_throughput(model, size, batch_size, device, model_args={}):
    n_channels, w, h = size
    net = MODELS[model]({'input_channels': n_channels, 'num_classes': 3, "output_channels": 3, **model_args}).to(device)
    dummy_input = torch.randn(batch_size, n_channels, w, h, dtype=torch.float).to(device)
    repetitions = 100
    warmup = 50
    total_time = 0
    with torch.no_grad():
        for rep in range(warmup):
            _ = net(dummy_input)
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = net(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    throughput = (repetitions * batch_size) / total_time
    print('Final Throughput:', throughput)


def seg_inference_inference_per_second(model, size, batch_size, device, model_args={}):
    n_channels, w, h = size
    net = MODELS[model]({'input_channels': n_channels, 'num_classes': 3, "output_channels": 3, **model_args}).to(device)
    dummy_input = torch.randn(batch_size, n_channels, w, h, dtype=torch.float).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 500
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(50):
        _ = net(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = net(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(f"Mean inference time: {mean_syn} ms")
    print(f"Time per example {mean_syn / batch_size} ms")


if __name__ == '__main__':
    models = [
        # ('lawin', 1, {'backbone_pretrained': True}),
        # ('lawin', 2, {'backbone_pretrained': True}),
        # ('lawin', 3, {'backbone_pretrained': True}),
        # ('lawin', 4, {'backbone_pretrained': True, 'main_pretrained': ['R', 'G', 'G', 'G']}),
        #
        # ('lawin', 1, {'backbone': 'MiT-B1'}),
        # ('lawin', 2, {'backbone': 'MiT-B1'}),
        ('lawin', 3, {'backbone': 'MiT-B1'}),
        # ('lawin', 4, {'backbone': 'MiT-B1'}),
        #
        # ('laweed', 1, {'backbone_pretrained': True}),
        # ('laweed', 2, {'backbone_pretrained': True}),
        # ('laweed', 3, {'backbone_pretrained': True}),
        # ('laweed', 4, {'backbone_pretrained': True, 'main_pretrained': ['R', 'G', 'G', 'G']}),
        #
        # ('laweed', 1, {'backbone_pretrained': True, 'backbone': 'MiT-B1'}),
        # ('laweed', 2, {'backbone_pretrained': True, 'backbone': 'MiT-B1'}),
        # ('laweed', 3, {'backbone_pretrained': True, 'backbone': 'MiT-B1'}),
        # ('laweed', 4, {'backbone_pretrained': True, 'main_pretrained': ['R', 'G', 'G', 'G'], 'backbone': 'MiT-B1'}),
        #
        #
        # ('splitlawin', 3, {'main_channels': 2}),
        # ('splitlawin', 4, {'main_channels': 2}),
        # ('splitlawin', 3, {'main_channels': 2, 'backbone': 'MiT-B1'}),
        # ('splitlawin', 4, {'main_channels': 2, 'backbone': 'MiT-B1'}),
        #
        # ('splitlaweed', 3, {'main_channels': 2, 'main_pretrained': ['R', 'G'], 'side_pretrained': 'G'}),
        # ('splitlaweed', 4, {'main_channels': 2, 'main_pretrained': ['R', 'G'], 'side_pretrained': 'G'}),
        #
        # ('doublelawin', 3, {'main_channels': 2}),
        # ('doublelawin', 4, {'main_channels': 2}),
        # ('doublelawin', 3, {'main_channels': 2, 'backbone': 'MiT-B1'}),
        # ('doublelawin', 4, {'main_channels': 2, 'backbone': 'MiT-B1'}),
        #
        # ('doublelaweed', 3, {'main_channels': 2, 'main_pretrained': ['R', 'G'], 'side_pretrained': 'G'}),
        # ('doublelaweed', 4, {'main_channels': 2, 'main_pretrained': ['R', 'G'], 'side_pretrained': 'G'}),
        # ('segnet', 1, {}),
        # ('segnet', 2, {}),
        # ('segnet', 3, {}),
        # ('segnet', 4, {}),
        ('resnet', 3, {"model_name": "50"}),
    ]
    per_layer_stats = False
    verbose = False
    batch_size = 32
    wh = (512, 512)
    for model, channels, args in models:
        with torch.cuda.device(0):
            size = (channels, ) + wh
            print(f"Model: {model}")
            print(f"Size: {size}")
            # seg_model_flops(model, channels, verbose, per_layer_stats, args)
            # seg_inference_throughput(model, channels, batch_size, 'cuda', args)
            seg_inference_inference_per_second(model, size, batch_size, 'cuda', args)
            torch.cuda.empty_cache()
            print()