from torchreid import models, utils


def show_available_model():
    models.show_avai_models()


def compute_complexity(model_list):
    param_dict = {}
    flops_dict = {}
    for model_name in model_list:
        try:
            model = models.build_model(name=model_name, num_classes=1000, pretrained=False)
            num_params, flops = utils.compute_model_complexity(model, (1, 3, 256, 128))
            # print(model_name + "\t\t\t\t\t\tParameters: " + str(num_params) + "\t\t\tFLOPs:" + str(flops))

            param_dict[model_name] = num_params
            flops_dict[model_name] = num_params

            # show detailed complexity for each module
            # utils.compute_model_complexity(model, (1, 3, 256, 128), verbose=True)

            # count flops for all layers including ReLU and BatchNorm
            # utils.compute_model_complexity(model, (1, 3, 256, 128), verbose=True, only_conv_linear=False)
        except Exception as e:
            print(model_name + "\tmodel not match the input", e)

    return param_dict, flops_dict


def sort_dict_show(string, dictionary):
    sorted_param_dict = dict(sorted(dictionary.items(), key=lambda item: item[1]))

    for item in sorted_param_dict:
        print(str(item) + "\t\t\t\t\t" + string + ": " + str(sorted_param_dict[item]))
