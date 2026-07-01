from .pansharpening_metrics import (
    PansharpeningMetrics,
    cross_correlation,
    D_lambda_torch,
    D_lambda_khan,
    D_s_torch,
    full_ref_metrics,
    hqnr,
    no_ref_metrics as _no_ref_metrics,
    psnr_ms,
    QIndex_torch,
    scc,
    sweep_no_ref_metrics,
)


def no_ref_metrics(lr_ms, pan, fused, fused_lr_override=None, ratio=4):
    try:
        return _no_ref_metrics(lr_ms, pan, fused, fused_lr_override=fused_lr_override, ratio=ratio)
    except TypeError:
        return _no_ref_metrics(lr_ms, pan, fused)
