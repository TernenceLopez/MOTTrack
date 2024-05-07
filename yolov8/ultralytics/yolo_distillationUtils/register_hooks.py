class HookTool:
    def __init__(self):
        self.fea = None

    def hook_fun(self, module, fea_in, fea_out):
        self.fea = fea_out


def get_feas_by_hook(model):
    fea_hooks = []

    # for i in [13, 17, 20, 23]:  # yolov5注意力对齐特征层
    for i in [13, 20, 23]:
        m = model.model[i]
        cur_hook = HookTool()
        m.register_forward_hook(cur_hook.hook_fun)
        fea_hooks.append(cur_hook)

    return fea_hooks


def get_s_feas_by_hook(model):
    fea_hooks = []
    remove_handles = []

    for i in [12, 15, 18, 21]:  # yolov8注意力对齐特征层
        m = model.model[i]
        cur_hook = HookTool()
        remove_handle = m.register_forward_hook(cur_hook.hook_fun)
        fea_hooks.append(cur_hook)
        remove_handles.append(remove_handle)

    return fea_hooks, remove_handles


def get_t_feas_by_hook(model):
    fea_hooks = []
    remove_handles = []

    for i in [12, 15, 18, 21]:
        m = model.model[i]
        cur_hook = HookTool()
        remove_handle = m.register_forward_hook(cur_hook.hook_fun)
        fea_hooks.append(cur_hook)
        remove_handles.append(remove_handle)

    return fea_hooks, remove_handles
