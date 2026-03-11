"""Lightweight test for SVG reconstruction environment.

Tests: dependencies → SVG rasterization → scoring → dataset loading (single sample).
No GPU required, minimal memory usage.

Usage:
    python test_svg_env.py [--dataset_path /path/to/starvector_svg-icons-simple]
"""

import sys
import time

sys.path.insert(0, ".")


def test_dependencies():
    """Check all required packages."""
    print("=" * 60)
    print("1. Dependency Check")
    print("=" * 60)
    all_ok = True
    for name, imp in [
        ("cairosvg", "import cairosvg; v=cairosvg.__version__"),
        ("bs4", "from bs4 import BeautifulSoup; v='OK'"),
        ("lxml", "import lxml; v='OK'"),
        ("cv2", "import cv2; v=cv2.__version__"),
        ("skimage", "from skimage.metrics import structural_similarity; v='OK'"),
        ("PIL", "from PIL import Image; v='OK'"),
        ("numpy", "import numpy; v=numpy.__version__"),
    ]:
        try:
            loc = {}
            exec(imp, {}, loc)
            print(f"  [OK] {name}: {loc.get('v', 'OK')}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            all_ok = False
    return all_ok


def test_rasterization():
    """Test SVG cleaning + rasterization with synthetic SVGs."""
    print("\n" + "=" * 60)
    print("2. SVG Rasterization")
    print("=" * 60)

    from roll.agentic.env.svg_reconstruction.svg_utils import process_and_rasterize_svg, pil_to_numpy
    import numpy as np

    test_svgs = {
        "circle": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="black"/></svg>',
        "rect": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect x="10" y="10" width="80" height="80" fill="red"/></svg>',
        "complex": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect width="100" height="100" fill="white"/><circle cx="30" cy="30" r="15" fill="blue"/><circle cx="70" cy="70" r="15" fill="green"/><line x1="30" y1="30" x2="70" y2="70" stroke="black" stroke-width="2"/></svg>',
    }

    all_ok = True
    for name, svg in test_svgs.items():
        t0 = time.time()
        pil_img = process_and_rasterize_svg(svg, resolution=256, timeout=5.0)
        elapsed = time.time() - t0
        if pil_img is not None:
            np_img = pil_to_numpy(pil_img, 256)
            is_blank = np.all(np_img >= 250)
            print(f"  [{'BLANK!' if is_blank else 'OK'}] {name}: shape={np_img.shape}, mean={np_img.mean():.1f}, time={elapsed:.3f}s")
            if is_blank:
                all_ok = False
        else:
            print(f"  [FAIL] {name}: returned None, time={elapsed:.3f}s")
            all_ok = False
    return all_ok


def test_scoring():
    """Test scoring pipeline with known image pairs."""
    print("\n" + "=" * 60)
    print("3. Scoring Pipeline")
    print("=" * 60)

    from roll.agentic.env.svg_reconstruction.svg_utils import process_and_rasterize_svg, pil_to_numpy
    from roll.agentic.env.svg_reconstruction.scoring import calculate_total_score
    import numpy as np

    # Create GT image from SVG
    gt_svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="black"/></svg>'
    gt_pil = process_and_rasterize_svg(gt_svg, resolution=256)
    if gt_pil is None:
        print("  [FAIL] Cannot rasterize GT SVG!")
        return False
    gt_np = pil_to_numpy(gt_pil, 256)

    # Test A: GT vs GT
    scores = calculate_total_score(gt_np, gt_np)
    print(f"  Test A - GT vs GT (expect ~1.0):")
    for k, v in scores.items():
        print(f"    {k}: {v:.4f}")
    ok_a = scores["total_score"] >= 0.99

    # Test B: GT vs blank
    blank = np.full((256, 256, 3), 255, dtype=np.uint8)
    scores = calculate_total_score(gt_np, blank)
    print(f"  Test B - GT vs Blank (expect low):")
    for k, v in scores.items():
        print(f"    {k}: {v:.4f}")
    ok_b = scores["total_score"] < 0.9

    # Test C: GT vs similar shape
    similar_svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="35" fill="black"/></svg>'
    sim_pil = process_and_rasterize_svg(similar_svg, resolution=256)
    if sim_pil:
        sim_np = pil_to_numpy(sim_pil, 256)
        scores = calculate_total_score(gt_np, sim_np)
        print(f"  Test C - GT vs Similar Circle (expect high):")
        for k, v in scores.items():
            print(f"    {k}: {v:.4f}")

    # Test D: GT vs completely different
    diff_svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect x="0" y="0" width="50" height="50" fill="red"/></svg>'
    diff_pil = process_and_rasterize_svg(diff_svg, resolution=256)
    if diff_pil:
        diff_np = pil_to_numpy(diff_pil, 256)
        scores = calculate_total_score(gt_np, diff_np)
        print(f"  Test D - GT vs Different Shape (expect medium-low):")
        for k, v in scores.items():
            print(f"    {k}: {v:.4f}")

    if ok_a and ok_b:
        print("\n  Scoring OK!")
    return ok_a and ok_b


def test_dataset_single_sample(dataset_path):
    """Load dataset and test ONE real sample end-to-end (low memory)."""
    print("\n" + "=" * 60)
    print("4. Dataset Loading + Real Sample Test")
    print("=" * 60)

    import os
    import glob
    import pyarrow as pa
    import numpy as np
    from roll.agentic.env.svg_reconstruction.svg_utils import process_and_rasterize_svg, pil_to_numpy
    from roll.agentic.env.svg_reconstruction.scoring import calculate_total_score

    split_dir = os.path.join(dataset_path, "train")
    load_dir = split_dir if os.path.isdir(split_dir) else dataset_path
    arrow_files = sorted(glob.glob(os.path.join(load_dir, "*.arrow")))

    if not arrow_files:
        print(f"  [FAIL] No .arrow files in {load_dir}")
        return False

    print(f"  Arrow files: {[os.path.basename(f) for f in arrow_files]}")

    # Read just a few rows to avoid memory issues
    with open(arrow_files[0], "rb") as fh:
        reader = pa.ipc.open_stream(fh)
        table = reader.read_all()

    num_rows = table.num_rows
    cols = table.column_names
    print(f"  Total rows: {num_rows}, columns: {cols}")

    # Get SVG column name
    svg_col = "Svg" if "Svg" in cols else "svg" if "svg" in cols else None
    if svg_col is None:
        print(f"  [FAIL] No SVG column found! Columns: {cols}")
        return False

    # Test a few samples
    test_indices = [0, 1, 100, 1000, 5000]
    success = 0
    for idx in test_indices:
        if idx >= num_rows:
            continue
        svg_text = table.column(svg_col)[idx].as_py()

        t0 = time.time()
        pil_img = process_and_rasterize_svg(svg_text, resolution=256, timeout=5.0)
        elapsed = time.time() - t0

        if pil_img is not None:
            np_img = pil_to_numpy(pil_img, 256)
            is_blank = np.all(np_img >= 250)

            # Also test scoring: GT vs itself
            scores = calculate_total_score(np_img, np_img)

            status = "BLANK!" if is_blank else "OK"
            print(f"  [{status}] idx={idx}: svg_len={len(svg_text)}, mean={np_img.mean():.1f}, "
                  f"self_score={scores['total_score']:.4f}, time={elapsed:.3f}s")
            if not is_blank:
                success += 1
        else:
            print(f"  [FAIL] idx={idx}: svg_len={len(svg_text)}, rasterize=None, time={elapsed:.3f}s")

    # Clean up
    del table

    print(f"\n  Results: {success}/{len([i for i in test_indices if i < num_rows])} real samples rendered successfully")
    return success > 0


def test_action_parsing():
    """Test action parsing (answer tag extraction + SVG extraction)."""
    print("\n" + "=" * 60)
    print("5. Action Parsing")
    print("=" * 60)

    from roll.agentic.env.svg_reconstruction.env import SVGReconstructionEnv
    from roll.agentic.env.svg_reconstruction.config import SVGReconstructionEnvConfig

    env = SVGReconstructionEnv(SVGReconstructionEnvConfig())

    cases = [
        ("valid", '<answer><svg xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="40"/></svg></answer>', True),
        ("with_think", '<think>Let me think...</think><answer><svg xmlns="http://www.w3.org/2000/svg"><rect width="100" height="100"/></svg></answer>', True),
        ("no_answer_tag", '<svg xmlns="http://www.w3.org/2000/svg"><circle/></svg>', False),
        ("no_svg", '<answer>I cannot draw this</answer>', False),
        ("empty", '', False),
    ]

    all_ok = True
    for name, text, expect_svg in cases:
        result = env.parse_action(text)
        has_svg = result["action"] is not None
        ok = has_svg == expect_svg
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: expect_svg={expect_svg}, got_svg={has_svg}")
        if not ok:
            all_ok = False

    env.close()
    return all_ok


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path",
        default="/mnt/project_modelware_roce/zhaojian/liangsirui/weiyu/projects/local_roll_dev/roll_dev/data/datasets_disk/starvector_svg-icons-simple")
    args = parser.parse_args()

    print("SVG Reconstruction Environment - Lightweight Test")
    print("=" * 60)

    results = {}

    results["dependencies"] = test_dependencies()
    if not results["dependencies"]:
        print("\nDependency check failed, aborting.")
        sys.exit(1)

    results["rasterization"] = test_rasterization()
    results["scoring"] = test_scoring()
    results["action_parsing"] = test_action_parsing()
    results["dataset"] = test_dataset_single_sample(args.dataset_path)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\nALL TESTS PASSED!")
    else:
        print("\nSOME TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
