"""Stage-wise divergence locator: export the full graph with intermediate
outputs (monodepth disparity, encoder features, decoder features) and compare
each stage ONNX-vs-PyTorch to find where the graphs first disagree.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from export_full_model import build_predictor  # noqa: E402
from sharp3d import resolve_cache_dir  # noqa: E402


class _DebugWrapper(torch.nn.Module):
    """Expose every pipeline stage as an output."""

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, image, disparity_factor):
        p = self.predictor
        mo = p.monodepth_model(image)
        monodepth_disparity = mo.disparity
        df = disparity_factor[:, None, None, None]
        monodepth = df / monodepth_disparity.clamp(min=1e-4, max=1e4)
        monodepth, _ = p.depth_alignment(monodepth, None, mo.decoder_features)
        init_output = p.init_model(image, monodepth)
        image_features = p.feature_model(
            init_output.feature_input, encodings=mo.output_features)
        delta = p.prediction_head(image_features)
        g = p.gaussian_composer(delta=delta,
                                base_values=init_output.gaussian_base_values,
                                global_scale=init_output.global_scale)
        outs = [monodepth_disparity, monodepth]
        outs += list(mo.output_features)          # 5 encoder feature maps
        outs.append(mo.decoder_features)
        outs += [g.mean_vectors, g.singular_values, g.quaternions,
                 g.colors, g.opacities]
        return tuple(outs)


def main():
    device = torch.device("cuda")
    predictor = build_predictor(device, "quality")
    wrapper = _DebugWrapper(predictor)

    image = torch.randn(1, 3, 1536, 1536, device=device)
    df = torch.tensor([1.0], device=device, dtype=torch.float32)

    dbg_path = resolve_cache_dir() / "onnx" / "full_debug.onnx"
    with torch.no_grad():
        torch.onnx.export(wrapper, (image, df), str(dbg_path),
                          opset_version=17,
                          input_names=["image", "disparity_factor"],
                          output_names=[f"stage_{i}" for i in range(13)],
                          dynamo=False, do_constant_folding=True)
    print("导出完成", flush=True)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(dbg_path),
                                providers=["CUDAExecutionProvider"])
    img_np = np.clip(np.random.randn(1, 3, 1536, 1536).astype(np.float32) * 0.5 + 0.5, 0, 1)
    ort_outs = sess.run(None, {"image": img_np, "disparity_factor": np.array([1.0], np.float32)})

    with torch.no_grad():
        ref = wrapper(torch.from_numpy(img_np).to(device),
                      torch.from_numpy(df_np := np.array([1.0], np.float32)).to(device))

    names = ["monodepth_disparity", "monodepth", "enc0", "enc1", "enc2",
             "enc3", "enc4(lowres)", "decoder_features",
             "mean_vectors", "singular_values", "quaternions",
             "colors", "opacities"]
    for name, o, r in zip(names, ort_outs, ref):
        r_np = r.float().cpu().numpy()
        ad = np.abs(o - r_np)
        print(f"  {name:16s} max={ad.max():.3e}  mean={ad.mean():.3e}  "
              f"ref_absmax={np.abs(r_np).max():.3f}", flush=True)


if __name__ == "__main__":
    main()
