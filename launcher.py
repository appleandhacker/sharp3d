"""PyInstaller entry point for sharp3d GUI."""
import sys
import os
import multiprocessing

# Disable torch.compile in frozen builds (no inductor infrastructure available)
os.environ["SHARP3D_NO_COMPILE"] = "1"

# Ensure the bundled sharp3d package is importable
if getattr(sys, "frozen", False):
    bundle_dir = sys._MEIPASS
    sys.path.insert(0, bundle_dir)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    from sharp3d.gui.__main__ import main
    sys.exit(main())
