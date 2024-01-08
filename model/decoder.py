# Package imports
import torch
import torch.nn as nn
import tinycudann as tcnn


class ColorNet(nn.Module):
    def __init__(self, config, input_ch=4, geo_feat_dim=15, 
                hidden_dim_color=64, num_layers_color=3):
        super(ColorNet, self).__init__()    # 调用 nn.Module 的构造函数，传入子类构造函数的参数
        self.config = config
        self.input_ch = input_ch                    # 根据编码器输出维度确定
        self.geo_feat_dim = geo_feat_dim            # 传入 15
        self.hidden_dim_color = hidden_dim_color    # 传入 32
        self.num_layers_color = num_layers_color    # 传入 2

        self.model = self.get_model(config['decoder']['tcnn_network'])
    
    def forward(self, input_feat):
        # h = torch.cat([embedded_dirs, geo_feat], dim=-1)
        return self.model(input_feat)
    
    def get_model(self, tcnn_network=False):
        if tcnn_network:
            print('Color net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch + self.geo_feat_dim,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim_color,
                    "n_hidden_layers": self.num_layers_color - 1,
                },
                #dtype=torch.float
            )

        color_net =  [] # 最终要返回的【颜色解码网络】
        for l in range(self.num_layers_color):  # 颜色网络 有 2 层
            if l == 0:      # 第一层
                in_dim = self.input_ch + self.geo_feat_dim  # 编码器输出维度 + 15
            else:           # 第二层
                in_dim = self.hidden_dim_color
            
            if l == self.num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = self.hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

            if l != self.num_layers_color - 1:
                color_net.append(nn.ReLU(inplace=True))

        return nn.Sequential(*nn.ModuleList(color_net))

class SDFNet(nn.Module):
    def __init__(self, config, input_ch=3, geo_feat_dim=15, hidden_dim=64, num_layers=2):
        super(SDFNet, self).__init__()  # 调用 nn.Module 的构造函数，传入子类构造函数的参数
        self.config = config
        self.input_ch = input_ch            # 根据编码器输出维度确定
        self.geo_feat_dim = geo_feat_dim    # 传入 15
        self.hidden_dim = hidden_dim        # 传入 32
        self.num_layers = num_layers        # 默认为 2，也就是 2 层 MLP

        # 创建网络模型，参考 tcnn_network
        self.model = self.get_model(tcnn_network=config['decoder']['tcnn_network'])
    
    def forward(self, x, return_geo=True):
        out = self.model(x)

        if return_geo:  # return feature
            return out
        else:
            return out[..., :1]

    def get_model(self, tcnn_network=False):
        if tcnn_network:
            print('SDF net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch,
                n_output_dims=1 + self.geo_feat_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim,
                    "n_hidden_layers": self.num_layers - 1,
                },
                #dtype=torch.float
            )
        else:
            sdf_net = []    # 最终要返回的 sdf 网络
            '''
            线性层1
                in: ?       out: 32
            线性层2
                in: 32      out: 16
            ReLU层
            '''
            for l in range(self.num_layers):    # l 为 0 和 1
                # 配置输入维度
                if l == 0:                      # 第 1 层时
                    in_dim = self.input_ch      # 根据初始化传入的参数设置输入维度
                else:                           # 第 2 层时
                    in_dim = self.hidden_dim    # 输入维度设置为默认值 64
                
                # 配置输出维度
                if l == self.num_layers - 1:        # 不是第一层
                    out_dim = 1 + self.geo_feat_dim # 1 sdf值 + 15 几何特征 (论文中的 h)
                else:                               # 第 1 层时
                    out_dim = self.hidden_dim       # 默认值 64
                
                sdf_net.append(nn.Linear(in_dim, out_dim, bias=False))  # 加入两个线性层
                
                if l != self.num_layers - 1:    # 在最后一层加入 ReLU 层
                    sdf_net.append(nn.ReLU(inplace=True))

            return nn.Sequential(*nn.ModuleList(sdf_net))   # 构建网络并返回

class ColorSDFNet(nn.Module):   # 未使用
    '''
    Color grid + SDF grid
    '''
    def __init__(self, config, input_ch=3, input_ch_pos=12):
        super(ColorSDFNet, self).__init__()
        self.config = config
        self.color_net = ColorNet(config,           # 类实例化
                input_ch=input_ch+input_ch_pos, 
                geo_feat_dim=config['decoder']['geo_feat_dim'], 
                hidden_dim_color=config['decoder']['hidden_dim_color'], 
                num_layers_color=config['decoder']['num_layers_color'])
        self.sdf_net = SDFNet(config,               # 类实例化
                input_ch=input_ch+input_ch_pos,
                geo_feat_dim=config['decoder']['geo_feat_dim'],
                hidden_dim=config['decoder']['hidden_dim'], 
                num_layers=config['decoder']['num_layers'])

    # 允许颜色网络直接访问原始的颜色嵌入信息(ColorSDFNet的特点)        
    def forward(self, embed, embed_pos, embed_color):

        if embed_pos is not None:
            h = self.sdf_net(torch.cat([embed, embed_pos], dim=-1), return_geo=True) 
        else:
            h = self.sdf_net(embed, return_geo=True) 
        
        sdf, geo_feat = h[...,:1], h[...,1:]
        if embed_pos is not None:
            rgb = self.color_net(torch.cat([embed_pos, embed_color, geo_feat], dim=-1))
        else:
            rgb = self.color_net(torch.cat([embed_color, geo_feat], dim=-1))
        
        return torch.cat([rgb, sdf], -1)
    
class ColorSDFNet_v2(nn.Module):    # TUM Replica Azure iPhone synthetic Scannet 均使用该类实例化
    '''
    No color grid
    '''
    def __init__(self, config, input_ch=3, input_ch_pos=12):
        super(ColorSDFNet_v2, self).__init__()
        self.config = config
        self.color_net = ColorNet(
            config,                     # 类实例化
            input_ch=input_ch_pos,      # 默认 12，初始化传入参数根据编码器输出维度确定
            geo_feat_dim=config['decoder']['geo_feat_dim'],         # replica 15
            hidden_dim_color=config['decoder']['hidden_dim_color'], # replica 32
            num_layers_color=config['decoder']['num_layers_color']) # replica 2
        self.sdf_net = SDFNet(
            config,                             # 类实例化
            input_ch=input_ch + input_ch_pos,   # 默认 3 + 12 = 15
            geo_feat_dim=config['decoder']['geo_feat_dim'], # replica 15
            hidden_dim=config['decoder']['hidden_dim'],     # replica 32
            num_layers=config['decoder']['num_layers'])     # replica 2
            
    # 颜色信息是通过结合空间编码和几何特征来生成的，而不是直接从独立的颜色数据中提取(ColorSDFNet_v2的特点)
    def forward(self, embed, embed_pos):

        if embed_pos is not None:
            h = self.sdf_net(torch.cat([embed, embed_pos], dim=-1), return_geo=True) 
        else:
            h = self.sdf_net(embed, return_geo=True) 
        
        sdf, geo_feat = h[...,:1], h[...,1:]
        if embed_pos is not None:
            rgb = self.color_net(torch.cat([embed_pos, geo_feat], dim=-1))
        else:
            rgb = self.color_net(torch.cat([geo_feat], dim=-1))
        
        return torch.cat([rgb, sdf], -1)