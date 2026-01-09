
import os
import argparse
import json
import csv
import numpy as np
import h5py
import matplotlib
matplotlib.use("TkAgg")  # or 'Qt5Agg' if you have PyQt5 installed
import matplotlib.pyplot as plt

# ---------- Dataset conventions ----------
SEG_PREFIX = "robot0_"
CAMERA_KEYS = {
    "left":  "agentview_left",
    "right": "agentview_right",
    "hand":  "eye_in_hand",
}
SEG_TYPES = {"element"}  # you requested element only

# ---------- HDF5 helpers ----------
def list_demos(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        g = f.get("data")
        if g is None:
            raise KeyError("Group 'data' not found in HDF5.")
        demos = []
        for k in g.keys():
            if k.startswith("demo_"):
                try:
                    task = json.loads(g[k].attrs['ep_meta'])['lang']
                    # data['data']['demo_1236'].attrs['ep_meta']
                    demos.append((int(k.split("_", 1)[1]), task))
                except ValueError:
                    pass
    # return sorted(demos)
    demos = sorted(demos)
    # print(len(demos))
    demos = check_ooi_not_exist(h5_path, demos)
    return demos

def check_ooi_exist(h5_path, demos):
    pass



def get_ooi_ids(dataset_name, demo_id, num_demo=100):
    # ooi_anno_dir = f"/home/binhng/Workspace/robocasa/robocasa/datasets/regenerate/robocasa-30and100demos-7chosen-tasks-for-Binh/{num_demo}/ooi_anno/{dataset_name.split('.')[0]}/"
    ooi_anno_dir = f"/home/binhng/Workspace/robocasa/robocasa/datasets/regenerate/robocasa-30and100demos-6chosen-tasks-for-aLoc/{num_demo}/ooi_anno/{dataset_name.split('.')[0]}/"
    ooi_files = [
        f"{demo_id.replace('_', '')}_left_element_ooi.json",
        f"{demo_id.replace('_', '')}_right_element_ooi.json",
        f"{demo_id.replace('_', '')}_hand_element_ooi.json"
    ]

    all_ooi_ids = []
    for file in ooi_files:
        try:
            with open(ooi_anno_dir+file, 'r', encoding='utf-8') as f:
                all_ooi_ids += json.load(f)  # parse JSON content from file
        except:
            continue

    all_ooi_ids = list(set(all_ooi_ids))
    # print("all_ooi_ids:", all_ooi_ids)
    return all_ooi_ids

def check_ooi_not_exist(h5_path, demos):
    dataset_name = h5_path.split("/")[-1].replace(".hdf5", "")
    num_demos = 100 if "100" in h5_path else 30
    # ooi_anno_dir = "/home/binhng/Workspace/robocasa/robocasa/datasets/regenerate/robocasa-30demos-5chosen-tasks-element/ooi_anno/"
    output = []
    # print(demos)
    for demo in demos:
        ooi_ids = get_ooi_ids(dataset_name, f"demo_{demo[0]}", num_demos)
        if len(ooi_ids) == 0:
            output.append(demo)
    return output

def list_obs_keys(h5_path: str, demo_id: int):
    with h5py.File(h5_path, "r") as f:
        g = f.get(f"data/demo_{demo_id}/obs")
        if g is None:
            raise KeyError(f"'data/demo_{demo_id}/obs' not found.")
        return sorted(list(g.keys()))

def _seg_key(camera: str, seg_type: str = "element") -> str:
    cam = CAMERA_KEYS[camera]
    return f"{SEG_PREFIX}{cam}_segmentation_{seg_type}"

def _squeeze_channel(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return arr[..., 0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr

def read_mask(h5_path: str, demo_id: int, camera: str, seg_type: str = "element") -> np.ndarray:
    key = _seg_key(camera, seg_type)
    with h5py.File(h5_path, "r") as f:
        ds = f.get(f"data/demo_{demo_id}/obs/{key}")
        if ds is None:
            avail = list_obs_keys(h5_path, demo_id)
            raise KeyError(f"Dataset '{key}' not found under demo_{demo_id}/obs.\nAvailable:\n{avail}")
        arr = ds[()]
    arr = _squeeze_channel(arr)
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int32, copy=False)
    return arr  # (T,H,W) or (H,W)

# ---------- Stats for bbox ----------
def dominant_value(roi: np.ndarray, ignore_ids=None, topk=3):
    if roi.size == 0:
        return None, 0.0, []
    flat = roi.reshape(-1)
    if ignore_ids:
        flat = flat[~np.isin(flat, np.array(ignore_ids))]
        if flat.size == 0:
            return None, 0.0, []
    vals, counts = np.unique(flat, return_counts=True)
    order = np.argsort(-counts)
    vals, counts = vals[order], counts[order]
    total = flat.size
    hist = [(int(v), int(c), float(c/total)) for v, c in zip(vals, counts)]
    if topk > 0 and len(hist) > topk:
        hist = hist[:topk]
    mode = int(vals[0]) if len(vals) else None
    dom = float(counts[0] / total) if len(counts) else 0.0
    return mode, dom, hist

# ---------- RectangleSelector (version-safe) ----------
from matplotlib.widgets import RectangleSelector
from packaging import version
import matplotlib.patches

def make_rect_selector(ax, onselect):
    common = dict(useblit=True, button=[1], minspanx=2, minspany=2, spancoords='pixels')
    if version.parse(matplotlib.__version__) >= version.parse("3.5.0"):
        return RectangleSelector(
            ax, onselect, interactive=True,
            useblit=True,
            minspanx=2, minspany=2,
            spancoords='pixels',
            button=[1],
            # props=dict(edgecolor='yellow', facecolor='none', linewidth=1.5),
            # handle_props=dict(edgecolor='yellow', facecolor='yellow'),
            # **common
        )
    else:
        return RectangleSelector(ax, onselect, drawtype='box', **common)

# ---------- Annotator ----------
class MPLAnnotator:
    def __init__(self, mask, ignore_ids=None, topk=3, start=0, title_prefix=""):
        if mask.ndim == 2:
            mask = mask[None, ...]
        self.mask = mask
        self.T, self.H, self.W = mask.shape
        self.frame = max(0, min(start, self.T - 1))
        self.ignore_ids = ignore_ids or []
        self.topk = topk
        self.title_prefix = title_prefix

        # records & patch artists per frame
        self.bboxes = {t: [] for t in range(self.T)}
        self.patches = {t: [] for t in range(self.T)}
        self.selected_patch = None

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.rs = make_rect_selector(self.ax, self.onselect)
        self.cid_key  = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.cid_pick = self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self._draw_frame()

    def _title(self):
        prefix = (self.title_prefix + " | ") if self.title_prefix else ""
        return f"{prefix}Frame {self.frame}/{self.T-1} (drag: draw, click: select, Del: delete, u: undo, r/R: reset, s: save, close to continue)"

    def _colorize(self, frame2d):
        m = frame2d.astype(np.float32)
        mmin, mmax = float(m.min()), float(m.max())
        return (m - mmin) / (mmax - mmin) if mmax > mmin else np.zeros_like(m)

    def _draw_frame(self):
        self.ax.clear()
        self.ax.set_title(self._title(), fontsize=10)
        disp = self._colorize(self.mask[self.frame])
        self.ax.imshow(disp, cmap='turbo', vmin=0.0, vmax=1.0, interpolation='nearest')
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(self.H, 0)  # invert y

        # rebuild patches and labels
        self.patches[self.frame].clear()
        self.selected_patch = None
        for b in self.bboxes[self.frame]:
            rect = matplotlib.patches.Rectangle(
                (b["x0"], b["y0"]), b["w"], b["h"],
                linewidth=1.5, edgecolor='lime', facecolor='none'
            )
            rect.set_picker(True)
            self.ax.add_patch(rect)
            self.patches[self.frame].append(rect)
            label = f"id:{b['mode']} {int(b['dominance']*100)}%"
            tx, ty = b["x0"] + 3, max(0, b["y0"] - 3)
            self.ax.text(tx, ty, label, color='lime', fontsize=8,
                         ha='left', va='top', backgroundcolor=(0, 0, 0, 0.4))
        self.fig.canvas.draw_idle()

    def goto(self, t):
        t = max(0, min(t, self.T - 1))
        if t != self.frame:
            self.frame = t
            self._draw_frame()
            print(f"Switched to frame {self.frame}/{self.T-1}")

    def onselect(self, eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        x0, y0 = int(round(eclick.xdata)),  int(round(eclick.ydata))
        x1, y1 = int(round(erelease.xdata)), int(round(erelease.ydata))
        x0 = max(0, min(x0, self.W - 1)); x1 = max(0, min(x1, self.W - 1))
        y0 = max(0, min(y0, self.H - 1)); y1 = max(0, min(y1, self.H - 1))
        x0, x1 = sorted([x0, x1]); y0, y1 = sorted([y0, y1])
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
        print(f"[frame {self.frame}] ADD bbox=({x0},{y0})-({x1},{y1}) | mode={mode} dom={dom:.2%}")
        for v, c, r in hist:
            print(f"  val={v} count={c} ratio={r:.2%}")
        self._draw_frame()

    def on_pick(self, event):
        artist = event.artist
        if isinstance(artist, matplotlib.patches.Rectangle):
            if self.selected_patch is not None:
                self.selected_patch.set_edgecolor('lime')
                self.selected_patch.set_linewidth(1.5)
            self.selected_patch = artist
            artist.set_edgecolor('red')
            artist.set_linewidth(2.5)
            self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in ['right', 'd']:
            self.goto(self.frame + 1)
        elif event.key in ['left', 'a']:
            self.goto(self.frame - 1)
        elif event.key == 'u':
            if self.bboxes[self.frame]:
                removed = self.bboxes[self.frame].pop()
                print(f"[frame {self.frame}] UNDO {removed}")
                self._draw_frame()
        elif event.key == 'r':
            n = len(self.bboxes[self.frame])
            self.bboxes[self.frame].clear()
            print(f"[frame {self.frame}] RESET ({n} bboxes cleared)")
            self._draw_frame()
        elif event.key == 'R':
            total = sum(len(self.bboxes[t]) for t in range(self.T))
            for t in range(self.T):
                self.bboxes[t].clear()
            print(f"RESET ALL ({total} bboxes cleared)")
            self._draw_frame()
        elif event.key in ['delete', 'backspace']:
            if self.selected_patch is not None:
                xb, yb = self.selected_patch.get_xy()
                w, h = self.selected_patch.get_width(), self.selected_patch.get_height()
                x0, y0 = int(round(xb)), int(round(yb))
                w_i, h_i = int(round(w)), int(round(h))
                before = len(self.bboxes[self.frame])
                self.bboxes[self.frame] = [
                    b for b in self.bboxes[self.frame]
                    if not (b["x0"] == x0 and b["y0"] == y0 and b["w"] == w_i and b["h"] == h_i)
                ]
                after = len(self.bboxes[self.frame])
                print(f"[frame {self.frame}] DELETE selected (removed {before-after})")
                self._draw_frame()
        elif event.key == 's':
            if hasattr(self, "out_json"):
                self.save(self.out_json, self.out_json_ooi)
            # if hasattr(self, "out_json_ooi"):
            #     self.save_ooi(self.out_json_ooi)

    def collect_all(self):
        out = []
        for t in range(self.T):
            out.extend(self.bboxes[t])
        return out

    def save(self, out_json, out_json_ooi):
        all_bboxes = self.collect_all()
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_bboxes, f, indent=2)
        print(f"Saved {len(all_bboxes)} bboxes to: {out_json}")
        
        ooi_ids = []
        for box in all_bboxes:
            ooi_ids.append(box['mode'])
        ooi_ids = list(set(ooi_ids))
        with open(out_json_ooi, "w", encoding="utf-8") as f:
            json.dump(ooi_ids, f, indent=2)
        print(f"Saved OOI summary to: {out_json_ooi}")

            

    # def save_ooi(self, out_json_ooi):
    #     """
    #     Example 'object-of-interest' summary: for each frame, gather all bbox mode IDs.
    #     You can customize to your labeling schema.
    #     """
        
        
        
    #     per_frame = {}
    #     for t in range(self.T):
    #         modes = [b["mode"] for b in self.bboxes[t] if b["mode"] is not None]
    #         per_frame[t] = sorted(set(modes))
    #     with open(out_json_ooi, "w", encoding="utf-8") as f:
    #         json.dump(per_frame, f, indent=2)
    #     print(f"Saved OOI summary to: {out_json_ooi}")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=str, required=True, help="Path to RoboCasa HDF5.")
    ap.add_argument("--demo", type=int, help="Single demo_id (if not using --all).")
    ap.add_argument("--all", action="store_true", help="Annotate all demos in the file.")
    ap.add_argument("--demo-start", type=int, default=0, help="Start index in the demo list when using --all.")
    ap.add_argument("--camera-start", type=int, default=0, help="Start camera index (0: left, 1: right, 2: hand).")
    ap.add_argument("--ignore", type=int, nargs="*", default=None, help="IDs to ignore (e.g., 0).")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--start", type=int, default=0, help="Start frame index.")
    ap.add_argument("--outdir", type=str, default="./ooi_anno/", help="Output directory.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cameras = list(CAMERA_KEYS.keys())  # ["left","right","hand"]
    if args.all:
        demos = list_demos(args.h5)
        print(f"Found {len(demos)} demos: starting at index {args.demo-start if 'demo-start' in args else 0}")
        # for di, demo_id in enumerate(demos[args.demo_start:], start=args.demo_start):
        for di, demo in enumerate(demos[args.demo_start:], start=args.demo_start):  
            demo_id, task = demo[0], demo[1]
            for ci, cam in enumerate(cameras[args.camera_start:], start=args.camera_start):
                try:
                    mask = read_mask(args.h5, demo_id, cam, "element")
                except KeyError as e:
                    print(f"[WARN] {e}")
                    continue
                print(f"Annotating demo_{demo_id} camera={cam} shape={mask.shape}")
                base = f"demo{demo_id}_{cam}_element"
                ann = MPLAnnotator(
                    mask,
                    ignore_ids=args.ignore,
                    topk=args.topk,
                    start=args.start,
                    title_prefix=f"demo={demo_id} cam={cam} type=element\ntask:{task}"
                )
                ann.out_json = os.path.join(args.outdir, f"{base}_bboxes.json")
                ann.out_json_ooi = os.path.join(args.outdir, f"{base}_ooi.json")
                plt.show()  # close to proceed to next (demo, camera)
                plt.close(ann.fig)
            # reset camera_start after first demo
            args.camera_start = 0
    else:
        if args.demo is None:
            raise ValueError("Provide --demo or use --all.")
        for cam in cameras:
            try:
                mask = read_mask(args.h5, args.demo, cam, "element")
            except KeyError as e:
                print(f"[WARN] {e}")
                continue
            print(f"Annotating demo_{args.demo} camera={cam} shape={mask.shape}")
            base = f"demo{args.demo}_{cam}_element"
            ann = MPLAnnotator(
                mask,
                ignore_ids=args.ignore,
                topk=args.topk,
                start=args.start,
                title_prefix=f"demo={args.demo} cam={cam} type=element"
            )
            ann.out_json = os.path.join(args.outdir, f"{base}_bboxes.json")
            ann.out_json_ooi = os.path.join(args.outdir, f"{base}_ooi.json")
            plt.show()
            plt.close(ann.fig)

if __name__ == "__main__":
    main()


'''

python notebooks/tool_all.py \
  --h5 /home/binhng/Workspace/robocasa/robocasa/datasets/regenerate/robocasa-30and100demos-6chosen-tasks-for-aLoc/30/OpenDrawer.hdf5 \
  --outdir ./ooi_anno/OpenDrawer/ \
  --ignore 0 \
  --topk 3 \
  --all


'''