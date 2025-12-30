
import os
import json
import csv
import argparse
from typing import List, Optional, Dict, Tuple
import numpy as np
import h5py
import matplotlib
matplotlib.use("TkAgg")  # or 'Qt5Agg', 'Agg' (non-interactive), adjust as needed
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# ---------------- HDF5 helpers ----------------
SEG_PREFIX = "robot0_"
CAMERA_KEYS = {
    "left":  "agentview_left",
    "right": "agentview_right",
    "hand":  "eye_in_hand",
}
SEG_TYPES = {"instance", "element", "class"}

def list_obs_keys(h5_path: str, demo_id: int) -> List[str]:
    with h5py.File(h5_path, "r") as f:
        g = f.get(f"data/demo_{demo_id}/obs")
        if g is None:
            raise KeyError(f"'data/demo_{demo_id}/obs' not found.")
        return sorted(list(g.keys()))

def _seg_key(camera: str, seg_type: str) -> str:
    if camera not in CAMERA_KEYS:
        raise ValueError(f"Unknown camera='{camera}'. Choose from {list(CAMERA_KEYS.keys())}.")
    if seg_type not in SEG_TYPES:
        raise ValueError(f"Unknown seg_type='{seg_type}'. Choose from {SEG_TYPES}.")
    cam = CAMERA_KEYS[camera]
    return f"{SEG_PREFIX}{cam}_segmentation_{seg_type}"

def _squeeze_channel(arr: np.ndarray) -> np.ndarray:
    # (T,H,W,1) -> (T,H,W), (H,W,1) -> (H,W)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return arr[..., 0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr

def read_mask(h5_path: str, demo_id: int, camera: str, seg_type: str) -> np.ndarray:
    key = _seg_key(camera, seg_type)
    with h5py.File(h5_path, "r") as f:
        ds = f.get(f"data/demo_{demo_id}/obs/{key}")
        if ds is None:
            avail = list_obs_keys(h5_path, demo_id)
            raise KeyError(f"Dataset '{key}' not found under demo_{demo_id}/obs.\n"
                           f"Available keys include:\n{avail}")
        arr = ds[()]  # read to numpy
    arr = _squeeze_channel(arr)
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int32, copy=False)
    return arr  # (T,H,W) or (H,W)

# ---------------- mode/dominance ----------------
def dominant_value(roi: np.ndarray, ignore_ids: Optional[List[int]] = None, topk: int = 1):
    if roi.size == 0:
        return None, 0.0, []
    flat = roi.reshape(-1)
    if ignore_ids and len(ignore_ids) > 0:
        keep = ~np.isin(flat, np.array(ignore_ids))
        flat = flat[keep]
        if flat.size == 0:
            return None, 0.0, []
    vals, counts = np.unique(flat, return_counts=True)
    order = np.argsort(-counts)  # desc by count
    vals, counts = vals[order], counts[order]
    total = flat.size
    ratios = counts / total
    hist_sorted = [(int(v), int(c), float(r)) for v, c, r in zip(vals, counts, ratios)]
    if topk > 0 and len(hist_sorted) > topk:
        hist_sorted = hist_sorted[:topk]
    mode = int(vals[0]) if len(vals) else None
    dominance = float(ratios[0]) if len(ratios) else 0.0
    return mode, dominance, hist_sorted

# ---------------- Matplotlib annotator ----------------
class MPLAnnotator:
    def __init__(self, mask: np.ndarray, ignore_ids: Optional[List[int]] = None, topk: int = 3, start: int = 0):
        # Normalize to (T,H,W)
        if mask.ndim == 2:
            mask = mask[None, ...]
        self.mask = mask
        self.T, self.H, self.W = mask.shape
        self.frame = max(0, min(start, self.T - 1))
        self.ignore_ids = ignore_ids or []
        self.topk = topk

        # Annotations per frame
        self.bboxes: Dict[int, List[Dict]] = {t: [] for t in range(self.T)}

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_title(self._title(), fontsize=10)
        self.im = None
        self.text_overlays: List[matplotlib.text.Text] = []

        # Rectangle selector
        self.rs = RectangleSelector(
            self.ax, self.onselect,
            # drawtype='box', 
            useblit=True,
            button=[1],  # left mouse
            minspanx=2, minspany=2,
            spancoords='pixels',
            interactive=True,
            # interactive=False
        )

        # Keybindings
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Initial draw
        self._draw_frame()

    def _title(self):
        return f"Frame {self.frame}/{self.T-1} | "
        # You can append custom info here

    def _colorize(self, frame2d: np.ndarray):
        # Normalize to 0..1 for colormap display
        m = frame2d.astype(np.float32)
        mmin, mmax = float(m.min()), float(m.max())
        if mmax > mmin:
            disp = (m - mmin) / (mmax - mmin)
        else:
            disp = np.zeros_like(m)
        return disp

    def _clear_texts(self):
        for t in self.text_overlays:
            t.remove()
        self.text_overlays.clear()

    def _draw_bboxes(self):
        # Remove previous bbox artists by clearing and redrawing image
        # Simpler approach: redraw the image and re-add texts
        self._clear_texts()
        for b in self.bboxes[self.frame]:
            # Text near top-left of bbox
            label = f"id:{b['mode']} {int(b['dominance']*100)}%"
            tx = b["x0"] + 3
            ty = max(0, b["y0"] - 3)
            text_artist = self.ax.text(tx, ty, label, color='lime', fontsize=8,
                                       ha='left', va='top', backgroundcolor=(0,0,0,0.4))
            self.text_overlays.append(text_artist)
            # Draw rectangle using matplotlib patch
            rect = matplotlib.patches.Rectangle(
                (b["x0"], b["y0"]),
                b["w"], b["h"],
                linewidth=1.5, edgecolor='lime', facecolor='none'
            )
            self.ax.add_patch(rect)

    # def _draw_frame(self):
    #     self.ax.clear()
    #     self.ax.set_title(self._title(), fontsize=10)
    #     disp = self._colorize(self.mask[self.frame])
    #     self.im = self.ax.imshow(disp, cmap='turbo', vmin=0.0, vmax=1.0, interpolation='nearest')
    #     self.ax.set_xlim(0, self.W)
    #     self.ax.set_ylim(self.H, 0)  # invert y-axis to match image coordinates
    #     self._draw_bboxes()
    #     self.fig.canvas.draw_idle()
            
            
        
    import matplotlib.patches  # ensure this import exists

    def _draw_frame(self):
        # Full redraw: clear axes, draw image, and re-add rectangles/text from records
        self.ax.clear()
        self.ax.set_title(self._title(), fontsize=10)

        disp = self._colorize(self.mask[self.frame])
        self.ax.imshow(disp, cmap='turbo', vmin=0.0, vmax=1.0, interpolation='nearest')
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(self.H, 0)  # invert y-axis to match image coords

        # Recreate rectangles and labels for all bboxes in the current frame
        for b in self.bboxes[self.frame]:
            rect = matplotlib.patches.Rectangle(
                (b["x0"], b["y0"]), b["w"], b["h"],
                linewidth=1.5, edgecolor='lime', facecolor='none'
            )
            self.ax.add_patch(rect)

            label = f"id:{b['mode']} {int(b['dominance']*100)}%"
            tx, ty = b["x0"] + 3, max(0, b["y0"] - 3)
            self.ax.text(tx, ty, label, color='lime', fontsize=8,
                        ha='left', va='top', backgroundcolor=(0, 0, 0, 0.4))

        self.fig.canvas.draw_idle()



    def goto(self, t: int):
        t = max(0, min(t, self.T - 1))
        if t != self.frame:
            self.frame = t
            self._draw_frame()
            print(f"Switched to frame {self.frame}/{self.T-1}")

    def onselect(self, eclick, erelease):
        # Coordinates are float; convert to int pixel bounds
        x0, y0 = int(round(eclick.xdata)), int(round(eclick.ydata))
        x1, y1 = int(round(erelease.xdata)), int(round(erelease.ydata))
        x0, x1 = sorted([max(0, min(x0, self.W-1)), max(0, min(x1, self.W-1))])
        y0, y1 = sorted([max(0, min(y0, self.H-1)), max(0, min(y1, self.H-1))])
        if x1 <= x0 or y1 <= y0:
            return
        roi = self.mask[self.frame, y0:y1+1, x0:x1+1]
        mode, dom, hist = dominant_value(roi, ignore_ids=self.ignore_ids, topk=self.topk)
        bbox = {
            "frame": int(self.frame),
            "x0": int(x0), "y0": int(y0),
            "x1": int(x1), "y1": int(y1),
            "w": int(x1 - x0 + 1), "h": int(y1 - y0 + 1),
            "mode": mode,
            "dominance": dom,
            "hist_topk": hist
        }
        self.bboxes[self.frame].append(bbox)
        print(f"[frame {self.frame}] bbox=({x0},{y0})-({x1},{y1}) | mode={mode} dom={dom:.2%}")
        for v, c, r in hist:
            print(f"  val={v} count={c} ratio={r:.2%}")
        self._draw_frame()

    def on_key(self, event):
        if event.key in ['right', 'd']:
            self.goto(self.frame + 1)
        elif event.key in ['left', 'a']:
            self.goto(self.frame - 1)
        elif event.key == 'u':
            # undo last bbox on current frame
            if self.bboxes[self.frame]:
                self.bboxes[self.frame].pop()
                self._draw_frame()
        elif event.key == 'r':
            # reset current frame
            self.bboxes[self.frame].clear()
            self._draw_frame()
        elif event.key == 'R':
            # reset all frames
            for t in range(self.T):
                self.bboxes[t].clear()
            self._draw_frame()
        elif event.key == 's':
            # save annotations
            print("Press Ctrl+C in terminal if you want to abort save.")
            # Save paths are set by main() via attributes
            self.save(self.out_json, self.out_json_ooi) #, self.out_csv)

    def save(self, out_json: str, out_json_ooi: str):
        all_bboxes = []
        for t in range(self.T):
            all_bboxes.extend(self.bboxes[t])
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_bboxes, f, indent=2)
        with open(out_json_ooi, "w", encoding="utf-8") as f:
            ooi_ids = []
            for box in all_bboxes:
                ooi_ids.append(box['mode'])
            ooi_ids = list(set(ooi_ids))
            json.dump(ooi_ids, f, indent=2)            
        # with open(out_csv, "w", newline="", encoding="utf-8") as f:
        #     writer = csv.DictWriter(
        #         f, fieldnames=["frame","x0","y0","x1","y1","w","h","mode","dominance","hist_topk"]
        #     )
        #         # write header
        #     writer.writeheader()
        #     for b in all_bboxes:
        #         writer.writerow(b)
        print(f"Saved {len(all_bboxes)} bboxes to:\n  {out_json}\n") # {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=str, required=True, help="Path to RoboCasa HDF5.")
    ap.add_argument("--demo", type=int, required=True, help="demo_id (e.g., 1688).")
    ap.add_argument("--camera", type=str, default="left", choices=list(CAMERA_KEYS.keys()))
    ap.add_argument("--type", type=str, default="element", choices=list(SEG_TYPES))
    ap.add_argument("--ignore", type=int, nargs="*", default=None, help="IDs to ignore (e.g., 0).")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="./ooi_anno/")
    args = ap.parse_args()

    mask = read_mask(args.h5, args.demo, args.camera, args.type)
    print(f"Loaded mask with shape {mask.shape}")

    ann = MPLAnnotator(mask, ignore_ids=args.ignore, topk=args.topk, start=args.start)

    # Set save paths
    os.makedirs(args.outdir, exist_ok=True)
    base = f"demo{args.demo}_{args.camera}_{args.type}"
    ann.out_json = os.path.join(args.outdir, f"{base}_bboxes.json")
    ann.out_json_ooi = os.path.join(args.outdir, f"{base}_ooi.json")
    # ann.out_csv  = os.path.join(args.outdir, f"{base}_bboxes.csv")

    plt.show()

if __name__ == "__main__":
    main()

'''

python notebooks/tool2.py \
  --h5 /home/binhng/Workspace/robocasa/robocasa/datasets/regenerate/robocasa-30demos-5chosen-tasks/CoffeeSetupMug.hdf5 \
  --demo 1688 \
  --camera left \
  --type element \
  --ignore 0 \
  --topk 3

'''