# 本文档记录 coslam.py 运行的流程

1. 首先进入 [coslam.py](./coslam.py) 文件783行 main 函数
   1. 调用 argparse 库读取参数。最终读取到的参数保存在变量`args`里。
   2. 根据`arg`中的地址，调用 [config.py](./config.py) 文件中的`load_config`方法，读取config文件，保存在变量`cfg`里。
   3. 
   4. 实例化 `Coslam` 类，传入参数为`cfg`。实例化对象为`slam`。
      1. 将传入的`cfg`参数保存为对象的属性。
      2. 用`torch.device`获取运行设备的类型。
      3. 根据`cfg`调用`get_dataset`方法实例化数据集类。
      4. 调用`self.create_bounds()`类方法，获取建图的边界。
      5. 调用`self.create_pose_data()`类方法，
   5. 调用类方法`slam.run()`。
