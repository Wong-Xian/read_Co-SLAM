import yaml


def load_config(path, default_path=None):
    """
    加载 config 文件
    参数：
        path (str): config 文件的路径
        default_path (str, 可选): 是否使用默认地址。默认无
    返回值：
        cfg: 字典数据类型
    """

    with open(path, 'r') as f:                      # 以读方式打开传入的参数 path
        cfg_special = yaml.full_load(f)             # 用 yaml 库加载文件中的所有配置信息

    inherit_from = cfg_special.get('inherit_from')  # 获取 inherit_from 信息

    if inherit_from is not None:                    # 有 inherit_from 信息
        cfg = load_config(inherit_from, default_path)# 加载父配置信息
    elif default_path is not None:                  # 有 default_path
        with open(default_path, 'r') as f:
            cfg = yaml.full_load(f)                 # 加载 default_path 配置信息
    else:
        cfg = dict()

    update_recursive(cfg, cfg_special)              # 把 cfg_special 合并到 cfg 中

    return cfg


def update_recursive(dict1, dict2):
    """
    把字典2中的数据合并到字典1中
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v