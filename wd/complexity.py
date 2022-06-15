import numpy as np
import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
from models import MODELS


def seg_model_flops(model, n_channels, verbose=False, per_layer_stats=False, model_args={}):

    net = MODELS[model]({'input_channels': n_channels, 'num_classes': 3, **model_args})
    macs, params = get_model_complexity_info(net, (n_channels, 256, 256), as_strings=True,
                                             print_per_layer_stat=per_layer_stats, verbose=verbose)
    print(model)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def seg_inference_throughput(model, n_channels, batch_size, device, model_args={}):
    net = MODELS[model]({'input_channels': n_channels, 'num_classes': 3, **model_args}).to(device)
    dummy_input = torch.randn(batch_size, n_channels, 256, 256, dtype=torch.float).to(device)
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


def seg_inference_inference_per_second(model, n_channels, batch_size, device, model_args={}):
    net = MODELS[model]({'input_channels': n_channels, 'num_classes': 3, **model_args}).to(device)
    dummy_input = torch.randn(batch_size, n_channels, 256, 256, dtype=torch.float).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
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
    channels = 3
    models = [
        ('lawin', {}),
        ('laweed', {}),
        ('splitlawin', {'main_channels': 2}),
        ('splitlaweed', {'main_channels': 2}),
        ('doublelawin', {'main_channels': 2}),
        ('doublelaweed', {'main_channels': 2}),
    ]
    per_layer_stats = False
    verbose = False
    batch_size = 8
    for model, args in models:
        with torch.cuda.device(0):
            seg_model_flops(model, channels, verbose, per_layer_stats, args)
            seg_inference_throughput(model, channels, batch_size, 'cuda', args)
            seg_inference_inference_per_second(model, channels, batch_size, 'cuda', args)
            torch.cuda.empty_cache()
            print()