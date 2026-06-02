from sgcr.sr_model import ShadowFormer_OmniSR
import torch.nn as nn
import torch


class NStageShadowFormer(nn.Module):
    """
    N-stage wrapper with optional weight tying.

    - tie_weights=False: instantiate N independent ShadowFormer_OmniSR modules.
    - tie_weights=True:  instantiate ONE ShadowFormer_OmniSR and reuse it for all stages.

    Chaining (opt.multi_stage_mode):
      - "direct":        y_{k+1} = f(y_k)
      - "residual":      y_{k+1} = y_k + f(y_k)
      - "full_residual": y_{k+1} = x   + f(y_k)

    Side effects:
      - self.ys: list of outputs per stage/iteration
      - self.y1/self.y2 kept for backwards compatibility when n_stage>=2
    """
    def __init__(
        self,
        opt,
        n_stage: int = 2,
        tie_weights: bool = False,
    ):
        super().__init__()
        if n_stage < 1:
            raise ValueError(f"n_stage must be >= 1, got {n_stage}")

        self.opt = opt
        self.n_stage = int(n_stage)
        self.tie_weights = bool(tie_weights)

        # Outputs (filled on forward)
        self.ys = []
        self.y1 = None
        self.y2 = None

        # For aux snapshots (filled on forward)
        self._aux_per_iter = []

        if self.tie_weights:
            # One shared module reused N times
            self.stage = ShadowFormer_OmniSR(opt)

            # NOTE: In your original 2-stage residual mode, you zero-init ONLY stage2 head.
            # With tied weights, you cannot do that without also changing stage1 behavior.
            # So we skip that special init in tie_weights mode.
        else:
            # N independent stages
            self.stages = nn.ModuleList([
                ShadowFormer_OmniSR(opt)
                for _ in range(self.n_stage)
            ])

            # Generalize your original residual-mode init:
            # zero-init heads of stages[1:] so they start near-identity when used as residual refiners.
            if opt.multi_stage_mode == "residual" and self.n_stage >= 2:
                self._zero_init_heads(stage_indices=range(1, self.n_stage))

        # ---- Optional YCbCr correction head ----
        self.use_ycbcr_correction = opt.use_ycbcr_correction
        if self.use_ycbcr_correction:
            self.correction_head = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, 3, 1, 1)
            )

    def forward(
        self,
        x,
        DINO_Mat_features=None,
        point=None,
        normal=None,
        mask=None,
        **kwargs
    ):
        self.ys = []
        self._aux_per_iter = []
        self.y1 = None
        self.y2 = None

        mode = self.opt.multi_stage_mode

        def run_stage(inp):
            if self.tie_weights:
                y = self.stage(inp, DINO_Mat_features, point, normal, mask, **kwargs)
                aux = getattr(self.stage, "_shadow_aux", None)
            else:
                y = self.stages[i](inp, DINO_Mat_features, point, normal, mask, **kwargs)
                aux = getattr(self.stages[i], "_shadow_aux", None)
            return y, aux

        # ---- stage 0 (equivalent to your stage1): always "direct" application ----
        i = 0
        y, aux = run_stage(x)
        self.ys.append(y)
        if aux is not None:
            # snapshot per-iter aux (dict of tensors); avoids overwriting when weights are tied
            self._aux_per_iter.append({k: v for k, v in aux.items()})
        else:
            self._aux_per_iter.append(None)

        # ---- stages 1..n-1: chained refinements ----
        for i in range(1, self.n_stage):
            out, aux = run_stage(y)

            if mode == "direct":
                y = out
            elif mode == "residual":
                y = out + y
            elif mode == "full_residual":
                y = out + x
            else:
                raise ValueError('Please select "direct", "residual", or "full_residual" as the multi_stage_mode!')

            self.ys.append(y)
            if aux is not None:
                self._aux_per_iter.append({k: v for k, v in aux.items()})
            else:
                self._aux_per_iter.append(None)

        # Backwards-compat attributes for existing training code
        if len(self.ys) >= 1:
            self.y1 = self.ys[0]
        if len(self.ys) >= 2:
            self.y2 = self.ys[1]

        # ---- Optional structured correction on top of final output ----
        if self.use_ycbcr_correction:
            y = self._apply_ycbcr_correction(y)

        return y

    # --------------------------------------------------------
    # YCbCr Structured Correction (same as TwoStageShadowFormer)
    # --------------------------------------------------------
    def _rgb_to_ycbcr(self, x):
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        Y  = 0.299 * r + 0.587 * g + 0.114 * b
        Cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        Cr = 0.5 * r - 0.418688 * g - 0.081312 * b
        return Y, Cb, Cr

    def _ycbcr_to_rgb(self, Y, Cb, Cr):
        r = Y + 1.402 * Cr
        g = Y - 0.344136 * Cb - 0.714136 * Cr
        b = Y + 1.772 * Cb
        return torch.cat([r, g, b], dim=1)

    def _apply_ycbcr_correction(self, y2):
        Y2, Cb2, Cr2 = self._rgb_to_ycbcr(y2)
        delta = self.correction_head(y2)
        dY  = delta[:, 0:1]
        dCb = delta[:, 1:2]
        dCr = delta[:, 2:3]
        alpha = 1.0
        beta  = 0.3
        Yf  = Y2  + alpha * dY
        Cbf = Cb2 + beta  * dCb
        Crf = Cr2 + beta  * dCr
        return self._ycbcr_to_rgb(Yf, Cbf, Crf)

    def _zero_init_heads(self, stage_indices):
        """
        Zero-init output_proj.proj[0] conv for selected stages.
        Only used when tie_weights=False.
        """
        if self.tie_weights:
            return
        for idx in stage_indices:
            op = self.stages[idx].output_proj
            conv = op.proj[0]
            nn.init.zeros_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        print(f"[INFO] Zero-initialized output_proj.proj[0] for stages: {list(stage_indices)}")