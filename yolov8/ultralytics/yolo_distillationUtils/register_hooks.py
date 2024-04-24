class HookTool:
    def __init__(self):
        self.fea = None

    def hook_fun(self, module, fea_in, fea_out):
        self.fea = fea_out


def get_feas_by_hook(model):
    fea_hooks = []
    # for i in [2, 4, 6, 8, 13, 17, 20, 23]:
    # for i in [13, 17, 20, 23]:
    for i in [13, 20, 23]:
        m = model.model[i]
        cur_hook = HookTool()
        m.register_forward_hook(cur_hook.hook_fun)
        fea_hooks.append(cur_hook)

    return fea_hooks


def get_s_feas_by_hook(model):
    fea_hooks = []
    # for i in [13, 17, 20, 23]:
    # for i in [13, 20, 23]:
    for i in [13, 20, 23]:  # yolov5s_st_pruning.yaml
        m = model.model[i]
        cur_hook = HookTool()
        m.register_forward_hook(cur_hook.hook_fun)
        fea_hooks.append(cur_hook)

    return fea_hooks


def get_t_feas_by_hook(model):
    fea_hooks = []

    for i in [2, 4, 6, 8, 13, 17, 20, 23]:
        m = model.model[i]
        cur_hook = HookTool()
        m.register_forward_hook(cur_hook.hook_fun)
        fea_hooks.append(cur_hook)

    return fea_hooks