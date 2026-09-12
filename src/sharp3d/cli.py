"""CLI entry point for sharp3d."""

import argparse
import sys
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="sharp3d - Convert 2D images/videos to stereoscopic 3D (SBS)"
    )
    parser.add_argument("input", type=str, help="Input image or video path")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path (default: input_sbs.ext)")
    parser.add_argument("--ipd", type=float, default=0.063,
                        help="Inter-pupillary distance (default: 0.063)")
    parser.add_argument("--codec", type=str, default="h264",
                        choices=["h264", "h265", "av1"],
                        help="Video codec (default: h264)")
    parser.add_argument("--crf", type=int, default=26,
                        help="Video quality CRF (default: 26, lower=better)")
    parser.add_argument("--format", type=str, default="full_sbs",
                        choices=["full_sbs", "half_sbs", "full_tb", "half_tb",
                                 "cross", "anaglyph"],
                        help="Stereo output format (default: full_sbs)")
    parser.add_argument("--decompose", type=str, default="analytical",
                        choices=["analytical", "svd"],
                        help="Decomposition method (default: analytical)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--fp32", action="store_true",
                        help="Use FP32 instead of FP16")
    parser.add_argument("--depth", action="store_true",
                        help="Also output depth map (images only)")
    parser.add_argument("--hdr", action="store_true",
                        help="Force HDR10 (10-bit PQ) output. Auto-enabled for HDR input.")
    parser.add_argument("--keyframe-interval", type=int, default=1,
                        help="Run full SHARP prediction every Nth frame; "
                             "in-between frames reuse keyframe geometry with "
                             "refreshed colors (~Nx faster, slight geometry "
                             "lag on fast motion). Default: 1 (off)")

    args = parser.parse_args()

    # Range validation (argparse choices would print an unreadable list for
    # numeric ranges; explicit checks give one-line errors).
    if not (0 <= args.crf <= 51):
        parser.error(f"--crf 必须在 0-51 之间（收到 {args.crf}）")
    if args.ipd <= 0:
        parser.error(f"--ipd 必须为正数（收到 {args.ipd}）")
    if args.keyframe_interval < 1:
        parser.error(f"--keyframe-interval 必须 >= 1（收到 {args.keyframe_interval}）")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem
        suffix = input_path.suffix
        output_path = input_path.parent / f"{stem}_sbs{suffix}"

    # Detect input type
    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    is_video = input_path.suffix.lower() in video_exts

    if is_video and output_path.suffix.lower() not in video_exts:
        output_path = output_path.with_suffix(".mp4")

    # Import pipeline (deferred to allow --help without loading torch)
    from .pipeline import Sharp3DPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pipeline = Sharp3DPipeline(
        device=device,
        use_compile=not args.no_compile,
        use_fp16=not args.fp32,
        decompose_method=args.decompose,
        ipd=args.ipd,
    )

    if is_video:
        print(f"Processing video: {input_path.name}")
        print(f"Codec: {args.codec}, CRF: {args.crf}")

        def on_progress(frame_idx, total, fps):
            if frame_idx % 10 == 0 or frame_idx == total - 1:
                print(f"  Frame {frame_idx + 1}/{total}: {fps:.2f} fps")

        result = pipeline.process_video(
            input_path, output_path,
            codec=args.codec, crf=args.crf,
            format=args.format,
            hdr_output=True if args.hdr else None,
            keyframe_interval=args.keyframe_interval,
            progress_callback=on_progress,
        )
        print(f"\nDone! {result['n_frames']} frames in {result['total_elapsed']:.1f}s")
        print(f"Average: {result['avg_frame_time']:.3f}s/frame ({result['fps']:.2f} fps)")
        if result.get("hdr"):
            print("Output: {} (HDR10)".format(result['output_path']))
        else:
            print(f"Output: {result['output_path']}")

    else:
        print(f"Processing image: {input_path.name}")

        def on_status(msg):
            print(f"  {msg}")

        result = pipeline.process_image(
            input_path, output_path,
            output_depth=args.depth,
            format=args.format,
            progress_callback=on_status,
        )
        print(f"\nDone! {result['elapsed']:.3f}s ({result['fps']:.2f} fps)")
        print(f"Output: {result['output_path']} ({result['output_size'][0]}x{result['output_size'][1]})")
        if "depth_path" in result:
            print(f"Depth:  {result['depth_path']}")


if __name__ == "__main__":
    main()
