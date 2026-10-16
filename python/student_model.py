
import torch
import torch.nn as nn
import torch.nn.functional as F


ENC_CHANNELS       = [128, 256, 512, 1024]
ENC_BLOCKS         = [2, 3, 4, 3]
FEAT_DIM           = 1024
FEAT_SIZE          = 6

CORR_POOL_SIZE     = 4
CORR_FLAT_DIM      = CORR_POOL_SIZE ** 2 * CORR_POOL_SIZE ** 2 + 2
CUE_DIM            = CORR_FLAT_DIM + 1

POLICY_HIDDEN      = [512, 256, 128]
NUM_ACTIONS        = 4

PROJ_DIM           = 512


class DepthwiseSeparable(nn.Module):

    def __init__(self, in_ch, out_ch, kernel=3, stride=2):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, stride=stride,
                            padding=kernel // 2, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)))
        x = F.relu(self.bn2(self.pw(x)))
        return x


class TinyEncoder(nn.Module):

    def __init__(self, channels=None, blocks=None):
        super().__init__()
        if channels is None:
            channels = ENC_CHANNELS
        if blocks is None:
            blocks = ENC_BLOCKS
        c0, c1, c2, c3 = channels
        b0, b1, b2, b3 = blocks

        self.stem = nn.Sequential(
            nn.Conv2d(3, c0, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(c0, c1, b1)
        self.stage2 = self._make_stage(c1, c2, b2)
        self.stage3 = self._make_stage(c2, c3, b3)

    @staticmethod
    def _make_stage(in_ch, out_ch, n_blocks):
        layers = [DepthwiseSeparable(in_ch, out_ch, stride=2)]
        for _ in range(n_blocks - 1):
            layers.append(DepthwiseSeparable(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


class SimplifiedCorrelation(nn.Module):

    def __init__(self, feat_dim=None, feat_h=None, feat_w=None, pool_size=4):
        super().__init__()
        self.feat_dim = feat_dim or FEAT_DIM
        self.feat_h = feat_h or FEAT_SIZE
        self.feat_w = feat_w or FEAT_SIZE
        self.pool_size = pool_size
        self.n_positions = pool_size * pool_size
        self.pool = nn.AdaptiveAvgPool2d(pool_size)

    def forward(self, f_goal, f_obs):
        B = f_goal.size(0)
        ps = self.pool_size

        g_pooled = self.pool(f_goal)
        o_pooled = self.pool(f_obs)

        g = g_pooled.view(B, self.feat_dim, -1).permute(0, 2, 1)
        o = o_pooled.view(B, self.feat_dim, -1).permute(0, 2, 1)

        g_norm = F.normalize(g, dim=-1)
        o_norm = F.normalize(o, dim=-1)

        cross_corr = torch.bmm(g_norm, o_norm.permute(0, 2, 1))
        cross_flat = cross_corr.reshape(B, self.n_positions ** 2)

        left_idx = [i * ps for i in range(ps)]
        right_idx = [i * ps + (ps - 1) for i in range(ps)]

        g_left = g_norm[:, left_idx, :].mean(dim=1)
        g_right = g_norm[:, right_idx, :].mean(dim=1)
        o_left = o_norm[:, left_idx, :].mean(dim=1)
        o_right = o_norm[:, right_idx, :].mean(dim=1)

        lr_sim = torch.stack([
            (g_left * o_left).sum(dim=-1),
            (g_right * o_right).sum(dim=-1),
        ], dim=-1)

        cue = torch.cat([cross_flat, lr_sim], dim=-1)
        return cue


class TinyPolicy(nn.Module):

    def __init__(self, cue_dim=None, hidden=None, num_actions=None):
        super().__init__()
        if cue_dim is None:
            cue_dim = CUE_DIM
        if hidden is None:
            hidden = POLICY_HIDDEN
        if num_actions is None:
            num_actions = NUM_ACTIONS

        layers = []
        prev_dim = cue_dim
        for h in hidden:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(inplace=True)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, cue):
        return self.net(cue)


class StudentModel(nn.Module):

    def __init__(self, num_actions=None):
        super().__init__()
        if num_actions is None:
            num_actions = NUM_ACTIONS
        self.encoder = TinyEncoder()
        self.correlation = SimplifiedCorrelation()
        self.policy = TinyPolicy(num_actions=num_actions)

        self.corr_projector = nn.Linear(CORR_FLAT_DIM, PROJ_DIM)

    def encode(self, x):
        return self.encoder(x)

    def get_cue(self, obs, goal):
        f_obs = self.encoder(obs)
        f_goal = self.encoder(goal)
        return self.correlation(f_goal, f_obs)

    def forward(self, obs, goal, obs_proximity):
        corr_cue = self.get_cue(obs, goal)
        cue = torch.cat([corr_cue,
                         obs_proximity.unsqueeze(-1)], dim=-1)
        return self.policy(cue)

    def forward_with_cue(self, corr_cue, obs_proximity):
        cue = torch.cat([corr_cue,
                         obs_proximity.unsqueeze(-1)], dim=-1)
        return self.policy(cue)

    def project_cue(self, corr_cue):
        return self.corr_projector(corr_cue)

    def get_deploy_params(self):
        total = 0
        for name, p in self.named_parameters():
            if "corr_projector" not in name:
                total += p.numel()
        return total


def print_model_summary():
    model = StudentModel()

    enc = sum(p.numel() for p in model.encoder.parameters())
    corr = sum(p.numel() for p in model.correlation.parameters())
    pol = sum(p.numel() for p in model.policy.parameters())
    proj = sum(p.numel() for p in model.corr_projector.parameters())
    deploy = model.get_deploy_params()
    total = sum(p.numel() for p in model.parameters())

    print("Student Model Summary")
    print("=" * 55)
    print(f"  TinyEncoder:           {enc:>10,} params")
    print(f"  SimplifiedCorrelation: {corr:>10,} params")
    print(f"  TinyPolicy:            {pol:>10,} params")
    print(f"  corr_projector:        {proj:>10,} params  [distill-only]")
    print("=" * 55)
    print(f"  Total:                 {total:>10,}")
    print(f"  Deploy (no projector): {deploy:>10,}")
    print(f"  Deploy INT8:           ~{deploy // 1024} KB "
          f"({deploy / (1024*1024):.2f} MB)")
    print()
    print("Configuration:")
    print(f"  ENC_CHANNELS  = {ENC_CHANNELS}")
    print(f"  ENC_BLOCKS    = {ENC_BLOCKS}")
    print(f"  POLICY_HIDDEN = {POLICY_HIDDEN}")
    print(f"  CUE_DIM       = {CUE_DIM} (corr:{CORR_FLAT_DIM} + proximity:1)")


if __name__ == "__main__":
    print_model_summary()

    model = StudentModel()
    obs = torch.randn(2, 3, 96, 96)
    goal = torch.randn(2, 3, 96, 96)
    proximity = torch.rand(2)
    logits = model(obs, goal, proximity)
    print(f"\nForward pass: obs {obs.shape} + proximity {proximity.shape} → logits {logits.shape}")
    print(f"Logits: {logits}")

    corr_cue = model.get_cue(obs, goal)
    proj = model.project_cue(corr_cue)
    print(f"Correlation cue: {corr_cue.shape}")
    print(f"Projected cue:   {proj.shape}")
    print(f"forward_with_cue: {model.forward_with_cue(corr_cue, proximity).shape}")
