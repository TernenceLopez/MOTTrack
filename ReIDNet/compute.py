from utils import *

if __name__ == "__main__":
    # show_available_model()

    model_list = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                  'resnext50_32x4d', 'resnext101_32x8d', 'resnet50_fc512', 'se_resnet50',
                  'se_resnet50_fc512', 'se_resnet101', 'se_resnext50_32x4d', 'se_resnext101_32x4d',
                  'densenet121', 'densenet169', 'densenet201', 'densenet161', 'densenet121_fc512',
                  'inceptionresnetv2', 'inceptionv4', 'xception', 'resnet50_ibn_a',
                  'resnet50_ibn_b', 'nasnsetmobile', 'mobilenetv2_x1_0', 'mobilenetv2_x1_4',
                  'shufflenet', 'squeezenet1_0', 'squeezenet1_0_fc512', 'squeezenet1_1',
                  'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5',
                  'shufflenet_v2_x2_0', 'mudeep', 'resnet50mid', 'hacnn', 'pcb_p6',
                  'pcb_p4', 'mlfn', 'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25',
                  'osnet_ibn_x1_0', 'osnet_ain_x1_0', 'osnet_ain_x0_75', 'osnet_ain_x0_5', 'osnet_ain_x0_25']
    param_dict, flops_dict = compute_complexity(model_list)
    sort_dict_show("Parameters", param_dict)
    sort_dict_show("FLOPs", flops_dict)
