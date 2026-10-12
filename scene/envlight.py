import torch
import nvdiffrast.torch as dr

class EnvLight(torch.nn.Module):
    def __init__(self, resolution=1024):
        super().__init__()
        
        # 环境光的默认参数（硬编码）
        self.env_map_res = 1024  # 环境光贴图分辨率
        self.env_optimize_from = 0  # 从第 0 步开始优化
        self.env_optimize_until = 1_000_000_000  # 优化迭代次数上限
        self.envmap_lr = 0.01  # 学习率

        self.to_opengl = torch.tensor(
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]], 
            dtype=torch.float32, device="cuda"
        )

        # 初始化立方体贴图，大小由 `env_map_res` 控制
        self.base = torch.nn.Parameter(
            0.5 * torch.ones(6, self.env_map_res, self.env_map_res, 3, requires_grad=True)
        )
        
        # 设置优化器，用于优化环境光
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.envmap_lr, eps=1e-15)

    def capture(self):
        """捕获当前状态用于保存或检查点恢复"""
        return self.base, self.optimizer.state_dict()

    def restore(self, model_args, training_args=None):
        """从检查点恢复模型状态"""
        self.base, opt_dict = model_args
        if training_args is not None:
            self.optimizer.load_state_dict(opt_dict)
    
    def training_setup(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.envmap_lr, eps=1e-15)

    def forward(self, l):
        """前向计算，用于计算给定方向的光照值"""
        l = (l.reshape(-1, 3) @ self.to_opengl.T).reshape(*l.shape).contiguous()
        prefix = l.shape[:-1]
        if len(prefix) != 3:
            l = l.reshape(1, 1, -1, l.shape[-1])

        # 使用 nvdiffrast 采样纹理
        light = dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
        return light.view(*prefix, -1)
