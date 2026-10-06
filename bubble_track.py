"""
Tracking bubbles through the frames of one train (bubble_train_analysis.ipynb).

The detections of every frame (<run_dir>/shapes/*.json) are linked into tracks, one track per bubble:

1. Frame to frame: bubbles are matched by the overlap (IoU) of their outlines after moving the previous outline by
   the bubble's recent motion. The assignment is one-to-one (Hungarian), so a bubble can only continue once.
   Overlap makes no assumption about the shape: deformed, non-elliptical bubbles keep their identity as long as
   they change gradually from one frame to the next (886 ns).
2. Gaps: a bubble that is not found stays "lost" for up to `max_gap` frames. Meanwhile it can
   (a) pick up a low-score detection of the model at its expected position ("rescued"; the model's low-score
       detections are written to <run_dir>/candidates/ by bubble_io.draft_frames(candidate_thresh=...)), or
   (b) re-connect when it is detected again.
   Frames that are still missing are filled with an outline morphed between the outlines before and after the
   gap ("interpolated"; signed-distance interpolation, so arbitrary shapes morph smoothly).
3. Events: how each track starts and ends is classified from the overlap with neighbouring bubbles and the
   volume balance (volume ~ r_eq^3):
       end:   merged (absorbed by an overlapping bubble, volume conserved), occluded (hidden by an overlapping
              bubble that did not grow), left_view, dissolved (shrank before vanishing), vanished, end_of_train
       start: merger_product (new bubble formed by a merger), split, emerged (appeared overlapping an existing
              bubble that did not shrink), entered_view, appeared, start_of_train
   A track that breaks only because the bubble changed abruptly (one bubble ends, one starts in its place with a
   similar volume) is joined back together.

Output in <run_dir>/tracked/, laid out like a results folder so every bubble_stats function works on it:
    shapes/<frame>.json        all outlines (detected + rescued + interpolated) with track_id and source
    bubbles.csv, frames.csv    measurements as bubble_io.export_results, plus track_id / source (per bubble)
                               and n_detected / n_rescued / n_interpolated (per frame)
    tracks.csv                 one row per track (lifetime, events, size change, growth rate, motion)
    events.csv                 one row per start / end event
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage import draw, measure

import bubble_io as bio
import bubble_rcnn as br
from bubble_stats import C, INK, INK2, _end_labels, _label_x, _room_right, _series, _xaxis


@dataclass
class TrackConfig:
    max_gap: int = 6                 # frames a bubble may be missing and still keep its identity
    link_scale: float = 2.0          # outlines are compared on a grid of native px x this
    min_iou: float = 0.3             # frame-to-frame link: outline overlap (IoU) after motion compensation
    gap_min_iou: float = 0.3         # re-connecting after a gap
    gap_penalty: float = 0.03        # matching cost per missing frame (prefers bubbles seen more recently)
    small_r_px: float = 3.0          # bubbles smaller than this (native px) may also link by centre distance
    velocity_window: int = 4         # last observations used to estimate a bubble's motion
    max_speed_frac: float = 0.5      # expected motion per frame capped at this fraction of the radius
    use_candidates: bool = True      # rescue missed frames with low-score detections (<run_dir>/candidates/)
    rescue_min_iou: float = 0.5      # a low-score detection must overlap the expected outline this much
    duplicate_iou: float = 0.5       # low-score detections overlapping a detection this much are ignored
    merge_min_cover: float = 0.4     # fraction of an ending bubble covered by another one -> merged / occluded
    volume_tol: tuple = (0.65, 1.45)  # (after / before) volume ratio accepted as volume conserving
    merge_window: int = 10           # frames after a merger in which the merged bubble's volume is measured
    join_range: tuple = (0.5, 2.0)   # volume ratio for joining a track broken by an abrupt change
    split_max_drop: float = 0.85     # 'split' needs the parent's volume to drop below this fraction
    dissolve_shrink: float = 0.8     # radius shrank below this fraction of its recent maximum -> 'dissolved'
    dissolve_r_px: float = 2.5       # vanished without overlap and smaller than this (native px) -> 'dissolved'
    min_track_frames: int = 3        # tracks detected in fewer frames are 'transient' (mostly false positives):
                                     # no events, and they never count as a merger partner
    edge_px: float = 1.0             # outline within this distance of the image border touches the edge


END_EVENTS = ["merged", "occluded", "left_view", "dissolved", "vanished", "end_of_train", "transient"]
START_EVENTS = ["merger_product", "split", "emerged", "entered_view", "appeared", "start_of_train", "transient"]


# ----------------------------------------------------------------------------
# Observations (one bubble outline in one frame)
# ----------------------------------------------------------------------------
class Obs:
    """A bubble outline in frame k, rasterised as a cropped mask on the link grid (native px x s)."""
    __slots__ = ("k", "shape", "source", "score", "m", "r0", "c0", "n", "cy", "cx", "r", "edge", "tid")

    @property
    def vol(self):
        return float(self.n) ** 1.5          # ~ r^3 (projected area^1.5)

    @property
    def box(self):
        return self.r0, self.c0, self.r0 + self.m.shape[0], self.c0 + self.m.shape[1]


def make_obs(k, shape, source, native_hw, s, edge_px=1.0, score=np.nan) -> Optional[Obs]:
    o = br.shape_outline(shape)
    if len(o) < 3:
        return None
    p = br.native_to_model(o, s)
    H, W = int(round(native_hw[0] * s)), int(round(native_hw[1] * s))
    r0, r1 = max(int(np.floor(p[:, 0].min())), 0), min(int(np.ceil(p[:, 0].max())) + 1, H)
    c0, c1 = max(int(np.floor(p[:, 1].min())), 0), min(int(np.ceil(p[:, 1].max())) + 1, W)
    if r1 <= r0 or c1 <= c0:
        return None                          # entirely outside the image
    m = np.zeros((r1 - r0, c1 - c0), bool)
    rr, cc = draw.polygon(p[:, 0] - r0, p[:, 1] - c0, m.shape)
    m[rr, cc] = True
    if not len(rr):                          # tiny outline: keep the pixel under its centre
        y, x = np.clip(p.mean(0) - (r0, c0), 0, np.array(m.shape) - 1).round().astype(int)
        m[y, x] = True
        rr, cc = np.array([y]), np.array([x])
    ob = Obs()
    ob.k, ob.shape, ob.source, ob.score, ob.tid = k, shape, source, score, -1
    ob.m, ob.r0, ob.c0, ob.n = m, r0, c0, int(len(rr))
    ob.cy, ob.cx = br.model_to_native(np.array([rr.mean() + r0, cc.mean() + c0]), s)
    ob.r = np.sqrt(ob.n / np.pi) / s
    Hn, Wn = native_hw
    ob.edge = bool(o[:, 0].min() <= edge_px - 0.5 or o[:, 1].min() <= edge_px - 0.5
                   or o[:, 0].max() >= Hn - 0.5 - edge_px or o[:, 1].max() >= Wn - 0.5 - edge_px)
    return ob


def _inter(a: Obs, b: Obs, dy=0, dx=0) -> int:
    """Pixels shared by a (shifted by dy, dx link px) and b."""
    ar0, ac0 = a.r0 + dy, a.c0 + dx
    r0, r1 = max(ar0, b.r0), min(ar0 + a.m.shape[0], b.r0 + b.m.shape[0])
    if r1 <= r0:
        return 0
    c0, c1 = max(ac0, b.c0), min(ac0 + a.m.shape[1], b.c0 + b.m.shape[1])
    if c1 <= c0:
        return 0
    return int(np.count_nonzero(a.m[r0 - ar0:r1 - ar0, c0 - ac0:c1 - ac0] & b.m[r0 - b.r0:r1 - b.r0, c0 - b.c0:c1 - b.c0]))


def _boxes(obs, shifts=None):
    b = np.array([o.box for o in obs], float).reshape(-1, 4)
    if shifts is not None and len(b):
        b += np.c_[shifts, shifts]
    return b


def _box_overlap(a, b, pad=0):
    return ((a[:, None, 0] < b[None, :, 2] + pad) & (a[:, None, 2] + pad > b[None, :, 0])
            & (a[:, None, 1] < b[None, :, 3] + pad) & (a[:, None, 3] + pad > b[None, :, 1]))


def _velocity(obs: list, tcfg: TrackConfig):
    """Mean motion (native px / frame) over the last observations, capped relative to the radius."""
    w = obs[-tcfg.velocity_window:]
    if len(w) < 2 or w[-1].k == w[0].k:
        return 0.0, 0.0
    vy, vx = (w[-1].cy - w[0].cy) / (w[-1].k - w[0].k), (w[-1].cx - w[0].cx) / (w[-1].k - w[0].k)
    sp, vmax = np.hypot(vy, vx), max(0.5, tcfg.max_speed_frac * w[-1].r)
    return (vy * vmax / sp, vx * vmax / sp) if sp > vmax else (vy, vx)


# ----------------------------------------------------------------------------
# Linking
# ----------------------------------------------------------------------------
_BIG = 1e6


def _match(prev, dets, s, min_iou, small_r_px=0.0, penalty=None):
    """One-to-one matching of predicted outlines to detections (minimum total cost, cost = 1 - IoU + penalty).
    prev: list of (Obs, dy, dx) - last outline of a track and its expected shift (native px).
    min_iou, penalty: scalars or one value per track. Returns [(i, j, iou)]."""
    if not prev or not dets:
        return []
    min_iou = np.broadcast_to(np.asarray(min_iou, float), (len(prev),))
    penalty = np.zeros(len(prev)) if penalty is None else np.broadcast_to(np.asarray(penalty, float), (len(prev),))
    shifts = np.array([[round(dy * s), round(dx * s)] for _, dy, dx in prev], float)
    pb, db = _boxes([p[0] for p in prev], shifts), _boxes(dets)
    near = _box_overlap(pb, db, pad=2)
    rows, cols = np.nonzero(near.any(1))[0], np.nonzero(near.any(0))[0]
    if not len(rows):
        return []
    cost = np.full((len(rows), len(cols)), _BIG)
    for a, i in enumerate(rows):
        o, dy, dx = prev[i]
        iy, ix = int(shifts[i, 0]), int(shifts[i, 1])
        for b, j in enumerate(cols):
            if not near[i, j]:
                continue
            d = dets[j]
            inter = _inter(o, d, iy, ix)
            iou = inter / (o.n + d.n - inter)
            if iou >= min_iou[i]:
                cost[a, b] = 1 - iou + penalty[i]
            elif max(o.r, d.r) <= small_r_px:         # tiny bubbles: a 1 px shift already ruins the IoU
                dist = np.hypot(o.cy + dy - d.cy, o.cx + dx - d.cx)
                if dist <= max(1.0, 0.3 * (o.r + d.r)) and min(o.n, d.n) / max(o.n, d.n) >= 0.5:
                    iou = min_iou[i] - 0.01 * (1 + dist)
                    cost[a, b] = 1 - iou + penalty[i]
    ri, ci = linear_sum_assignment(cost)
    return [(int(rows[a]), int(cols[b]), 1 - cost[a, b] + penalty[rows[a]]) for a, b in zip(ri, ci) if cost[a, b] < _BIG]


def _not_duplicates(cands, dets, max_iou):
    if not cands or not dets:
        return list(cands)
    near = _box_overlap(_boxes(cands), _boxes(dets))
    keep = []
    for i, c in enumerate(cands):
        dup = False
        for j in np.nonzero(near[i])[0]:
            inter = _inter(c, dets[j])
            if inter / (c.n + dets[j].n - inter) > max_iou:
                dup = True
                break
        if not dup:
            keep.append(c)
    return keep


def link(dets_by_k, cands_by_k, tcfg: TrackConfig, s: float) -> dict:
    """Frame-by-frame linking with gap bridging. Returns {track_id: [Obs, ...]} (observed frames only)."""
    tracks, open_, nid = {}, [], 0
    for k, dets in enumerate(dets_by_k):
        matched, taken = set(), set()
        # 1. continue tracks seen in the previous frame or lost for up to max_gap frames - one joint assignment,
        #    with a small cost per missing frame (a strict priority for recently seen tracks lets a neighbour
        #    take the detection of an overlapping bubble that was missed for a frame)
        ages = np.array([k - tracks[t][-1].k for t in open_])
        prev = [(tracks[t][-1], *(v * a for v in _velocity(tracks[t], tcfg))) for t, a in zip(open_, ages)]
        pairs = _match(prev, dets, s, np.where(ages == 1, tcfg.min_iou, tcfg.gap_min_iou), tcfg.small_r_px,
                       penalty=tcfg.gap_penalty * (ages - 1))
        for i, j, _ in pairs:
            dets[j].tid = open_[i]
            tracks[open_[i]].append(dets[j])
            matched.add(open_[i])
            taken.add(j)
        free = [j for j in range(len(dets)) if j not in taken]
        # 2. rescue: tracks still unmatched may continue with a low-score detection at their expected position
        if cands_by_k is not None and cands_by_k[k]:
            T = [t for t in open_ if t not in matched]
            cands = _not_duplicates(cands_by_k[k], dets, tcfg.duplicate_iou) if T else []
            if T and cands:
                prev = [(tracks[t][-1], *(v * (k - tracks[t][-1].k) for v in _velocity(tracks[t], tcfg))) for t in T]
                for i, j, _ in _match(prev, cands, s, tcfg.rescue_min_iou):
                    c = cands[j]
                    c.tid, c.source = T[i], "rescued"
                    tracks[T[i]].append(c)
        # 3. everything else starts a new track
        for j in free:
            dets[j].tid = nid
            tracks[nid] = [dets[j]]
            open_.append(nid)
            nid += 1
        open_ = [t for t in open_ if k + 1 - tracks[t][-1].k <= tcfg.max_gap + 1]
    return tracks


# ----------------------------------------------------------------------------
# Gap filling (shape morphing)
# ----------------------------------------------------------------------------
def _sdf(m):
    return ndimage.distance_transform_edt(~m) - ndimage.distance_transform_edt(m)


def _centred(o: Obs, half):
    """o's mask on a (2*half+1)^2 canvas with its centroid at the centre."""
    rr, cc = np.nonzero(o.m)
    oy, ox = half[0] - int(round(rr.mean())), half[1] - int(round(cc.mean()))
    out = np.zeros((2 * half[0] + 1, 2 * half[1] + 1), bool)
    out[rr + oy, cc + ox] = True
    return out


def _ellipse_of(shape):
    """(xc, yc, a, b, theta) of a napari ellipse shape (a >= b, theta of the a-axis from x)."""
    d = np.asarray(shape["data"], float)
    c = d.mean(axis=0)
    e1, e2 = (d[1] - d[0]) / 2.0, (d[3] - d[0]) / 2.0        # (row, col) half-axis vectors
    if np.hypot(*e2) > np.hypot(*e1):
        e1, e2 = e2, e1
    return c[1], c[0], float(np.hypot(*e1)), float(np.hypot(*e2)), float(np.arctan2(e1[0], e1[1]))


def morph_shape(a: Obs, b: Obs, w: float, s: float) -> dict:
    """Outline between a (w=0) and b (w=1). Two circles / ellipses: their centre, axes and angle are interpolated
    (an ellipse again). Otherwise signed-distance interpolation of the two outlines (aligned at their centroids),
    placed at the interpolated centroid (a polygon)."""
    if a.shape["type"] == "ellipse" and b.shape["type"] == "ellipse":
        pa, pb = _ellipse_of(a.shape), _ellipse_of(b.shape)
        dth = (pb[4] - pa[4] + np.pi / 2) % np.pi - np.pi / 2          # axis angle is defined modulo pi
        p = [(1 - w) * x + w * y for x, y in zip(pa[:4], pb[:4])] + [pa[4] + w * dth]
        kind = "circle" if p[3] >= 0.999 * p[2] else "ellipse"
        return {**br._ellipse_shape(*p), "fit_kind": kind}
    half = [int(np.ceil(max(o.m.shape[d] for o in (a, b)))) + 3 for d in (0, 1)]
    f = (1 - w) * _sdf(_centred(a, half)) + w * _sdf(_centred(b, half))
    centre = br.native_to_model(np.array([(1 - w) * a.cy + w * b.cy, (1 - w) * a.cx + w * b.cx]), s)
    cs = measure.find_contours(f, 0.0) if f.min() < 0 else []
    if cs:
        c = max(cs, key=len)
        poly = br.model_to_native(c - np.array(half) + centre, s)
        poly = measure.approximate_polygon(poly, 0.2)[:-1]
        if len(poly) >= 4:
            return dict(type="polygon", data=poly.tolist())
    r = ((1 - w) * a.r + w * b.r)                 # degenerate: circle of the interpolated radius
    th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    cy, cx = (1 - w) * a.cy + w * b.cy, (1 - w) * a.cx + w * b.cx
    return dict(type="polygon", data=np.c_[cy + r * np.sin(th), cx + r * np.cos(th)].tolist())


def fill_gaps(tracks: dict, native_hw, s, tcfg: TrackConfig) -> int:
    n = 0
    for tid, obs in tracks.items():
        out = [obs[0]]
        for a, b in zip(obs[:-1], obs[1:]):
            for k in range(a.k + 1, b.k):
                o = make_obs(k, morph_shape(a, b, (k - a.k) / (b.k - a.k), s), "interpolated", native_hw, s, tcfg.edge_px)
                if o is not None:
                    o.tid = tid
                    out.append(o)
                    n += 1
            out.append(b)
        tracks[tid] = out
    return n


# ----------------------------------------------------------------------------
# Events
# ----------------------------------------------------------------------------
def _best_cover(o: Obs, others, dy, dx, s, exclude=-1, skip=frozenset()):
    """The observation in `others` covering the largest fraction of o (shifted by dy, dx native px).
    Observations of track `exclude` and of the tracks in `skip` are ignored."""
    best, bc = None, 0.0
    if not others:
        return best, bc
    iy, ix = round(dy * s), round(dx * s)
    ob = _boxes([o], np.array([[iy, ix]], float))
    near = _box_overlap(ob, _boxes(others))[0]
    for j in np.nonzero(near)[0]:
        d = others[j]
        if d.tid == exclude or d.tid in skip:
            continue
        c = _inter(o, d, iy, ix) / o.n
        if c > bc:
            best, bc = d, c
    return best, bc


def _covering(o: Obs, others, dy, dx, s, min_cover, exclude=-1, skip=frozenset()):
    """All observations in `others` covering at least min_cover of o (shifted): [(Obs, cover)]."""
    if not others:
        return []
    iy, ix = round(dy * s), round(dx * s)
    near = _box_overlap(_boxes([o], np.array([[iy, ix]], float)), _boxes(others))[0]
    out = []
    for j in np.nonzero(near)[0]:
        d = others[j]
        if d.tid == exclude or d.tid in skip:
            continue
        c = _inter(o, d, iy, ix) / o.n
        if c >= min_cover:
            out.append((d, c))
    return out


def _shrink(obs: list) -> float:
    """Recent radius relative to the largest (3-frame median) radius of the last 15 frames; < 1: shrinking."""
    r = pd.Series([o.r for o in obs[-15:]])
    if len(r) < 4:
        return 1.0
    return float(r.iloc[-3:].median() / r.rolling(3, center=True, min_periods=2).median().max())


def _by_frame(tracks, K):
    out = [[] for _ in range(K)]
    for obs in tracks.values():
        for o in obs:
            out[o.k].append(o)
    return out


def join_broken(tracks: dict, K: int, s, tcfg: TrackConfig) -> int:
    """Join track pairs where one bubble ends and a single new one starts in its place (similar volume) in the
    next frame: the same bubble whose outline changed too abruptly for the frame-to-frame link."""
    frames = _by_frame(tracks, K)
    end_at = {}
    for t, obs in tracks.items():
        end_at.setdefault(obs[-1].k, set()).add(t)
    n = 0
    for k in range(K - 1):
        choice = {}
        for t in sorted(end_at.get(k, ())):
            o = tracks[t][-1]
            d, cov = _best_cover(o, frames[k + 1], *_velocity(tracks[t], tcfg), s, exclude=t)
            if d is not None and cov >= tcfg.merge_min_cover:
                choice.setdefault(id(d), []).append((t, d))
        for group in choice.values():
            if len(group) != 1:
                continue
            t, d = group[0]
            u = d.tid
            if u == t or tracks[u][0] is not d:
                continue                       # d continues an existing bubble: merge / occlusion, not a break
            ratio = d.vol / tracks[t][-1].vol
            if not (tcfg.join_range[0] <= ratio <= tcfg.join_range[1]):
                continue
            for o in tracks[u]:
                o.tid = t
            end_at[tracks[u][-1].k].discard(u)
            end_at[tracks[u][-1].k].add(t)
            end_at[k].discard(t)
            tracks[t].extend(tracks.pop(u))
            n += 1
    return n


def classify_events(tracks: dict, K: int, s, tcfg: TrackConfig) -> dict:
    """{track_id: dict(start_event, start_partner, end_event, end_partner, end_volume_ratio, ...)}"""
    frames = _by_frame(tracks, K)
    at = {(o.tid, o.k): o for f in frames for o in f}
    info = {t: dict(start_event=None, start_partner=-1, start_volume_ratio=np.nan,
                    end_event=None, end_partner=-1, end_volume_ratio=np.nan) for t in tracks}
    transient = frozenset(t for t, obs in tracks.items()
                          if sum(o.source != "interpolated" for o in obs) < tcfg.min_track_frames)
    for t in transient:
        info[t].update(start_event="transient", end_event="transient")
    end_at = {}
    for t, obs in tracks.items():
        if t not in transient:
            end_at.setdefault(obs[-1].k, []).append(t)
    # --- ends ---
    for k in range(K):
        ending = sorted(end_at.get(k, ()))
        if k == K - 1:
            for t in ending:
                info[t]["end_event"] = "end_of_train"
            continue
        groups = {}
        for t in ending:
            o = tracks[t][-1]
            if not o.edge and _shrink(tracks[t]) <= tcfg.dissolve_shrink:   # shrank gradually before vanishing
                info[t]["end_event"] = "dissolved"                          # (at the edge: leaving looks the same)
                continue
            cover = _covering(o, frames[k + 1], *_velocity(tracks[t], tcfg), s, tcfg.merge_min_cover,
                              exclude=t, skip=transient)
            if cover:
                # the absorbing bubble is the one that grew the most (new bubbles: formed by the merger)
                def growth(dc):
                    prev = at.get((dc[0].tid, k))
                    return (dc[0].vol / prev.vol if prev is not None else np.inf, dc[1])
                d = max(cover, key=growth)[0]
                groups.setdefault(id(d), [d, []])[1].append(t)
                continue
            if o.edge:
                info[t]["end_event"] = "left_view"
            elif o.r <= tcfg.dissolve_r_px:
                info[t]["end_event"] = "dissolved"
            else:      # unexplained; within max_gap of the end it may just be missed in the last frames
                info[t]["end_event"] = "end_of_train" if K - 1 - o.k <= tcfg.max_gap else "vanished"
        for d, parents in groups.values():
            u = d.tid
            prev_u = at.get((u, k))                         # the absorbing bubble before the merger
            v_par = sum(np.median([o.vol for o in tracks[t][-3:]]) for t in parents)
            # a merged bubble first looks like the union of the two outlines and rounds up over the next frames,
            # so its volume is taken as the largest (3-frame median) value of the following merge_window frames
            after = pd.Series([at[(u, j)].vol for j in range(k + 1, k + 1 + tcfg.merge_window) if (u, j) in at])
            v_after = float(after.rolling(3, min_periods=1).median().max())
            v_before = np.median([at[(u, j)].vol for j in range(k - 2, k + 1) if (u, j) in at]) if prev_u else 0.0
            ratio = v_after / (v_par + v_before)
            edge = all(tracks[t][-1].edge for t in parents)
            if ratio < tcfg.volume_tol[0]:
                ev = "occluded"                             # the covering bubble is too small to contain both
            elif prev_u is None:
                ev = "merged"                               # a new bubble formed in their place
            else:
                # merged: the covering bubble grew by the parents' volume; occluded: it kept its volume
                err_merge, err_hidden = abs(np.log(ratio)), abs(np.log(v_after / v_before))
                if err_merge < err_hidden - 0.05:
                    ev = "merged"
                elif err_hidden < err_merge - 0.05:
                    ev = "occluded"
                elif K - 1 - k <= tcfg.max_gap:             # too small to tell, and maybe only missed at the end
                    ev = "end_of_train"
                else:                                       # too small to tell from the volume
                    ev = "occluded" if edge else "merged"
            for t in parents:
                e = "left_view" if ev == "occluded" and tracks[t][-1].edge else ev
                info[t].update(end_event=e, end_partner=u if e in ("merged", "occluded") else -1, end_volume_ratio=ratio)
            if ev == "merged" and prev_u is None and tracks[u][0] is d:
                info[u].update(start_event="merger_product", start_partner=parents[0], start_volume_ratio=ratio)
    # --- starts ---
    for t, obs in tracks.items():
        o = obs[0]
        if info[t]["start_event"] is not None:
            continue
        if o.k == 0:
            info[t]["start_event"] = "start_of_train"
            continue
        vy, vx = _velocity(obs[:tcfg.velocity_window], tcfg)
        p, cov = _best_cover(o, frames[o.k - 1], -vy, -vx, s, exclude=t, skip=transient)
        ev, partner, ratio = ("entered_view" if o.edge else "appeared"), -1, np.nan
        if p is not None and cov >= tcfg.merge_min_cover:
            q_now = at.get((p.tid, o.k))
            if q_now is not None:
                drop, ratio = q_now.vol / p.vol, (q_now.vol + o.vol) / p.vol
                split = drop <= tcfg.split_max_drop and tcfg.volume_tol[0] <= ratio <= tcfg.volume_tol[1]
                ev, partner = ("split" if split else "emerged"), p.tid
        if ev == "appeared" and o.k <= tcfg.max_gap:
            ev = "start_of_train"      # may just have been missed in the first frames
        info[t].update(start_event=ev, start_partner=partner, start_volume_ratio=ratio)
    return info


# ----------------------------------------------------------------------------
# Whole run
# ----------------------------------------------------------------------------
def tracked_dir(run_dir):
    return os.path.join(run_dir, "tracked")


def _image_hw(path):
    import tifffile
    with tifffile.TiffFile(path) as tf:
        return tuple(tf.pages[0].shape[:2])


def track_run(run_dir: str, cfg: br.RCNNConfig, tcfg: Optional[TrackConfig] = None, include_unreviewed=True,
              frame_interval_us=0.886, verbose=True):
    """Track all frames of a run (one train) and write <run_dir>/tracked/. Returns (bubbles, frames, tracks, events)."""
    tcfg = tcfg or TrackConfig()
    t0 = time.time()
    lst = bio.list_results(run_dir)
    if lst.empty:
        raise FileNotFoundError(f"no frames in {bio.shapes_dir(run_dir)}")
    if not include_unreviewed:
        lst = lst[lst["reviewed"]]
    if lst["train"].nunique() > 1:
        raise ValueError("tracking needs the frames of a single train; this run has trains "
                         f"{sorted(lst['train'].unique())}")
    lst = lst.sort_values("frame_idx").reset_index(drop=True)
    if verbose and len(lst) > 1 and (np.diff(lst["frame_idx"]) != 1).any():
        print("warning: frame numbers are not consecutive; gaps between files count as single frames")
    K, s = len(lst), tcfg.link_scale
    dets, cands, metas = [], [], []
    for k, r in lst.iterrows():
        with open(r.json_path) as f:
            a = json.load(f)
        hw = _image_hw(os.path.join(os.path.dirname(r.json_path), a["image"]))
        metas.append((r, a, hw))
        dets.append([o for o in (make_obs(k, sh, "detected", hw, s, tcfg.edge_px) for sh in a.get("bubbles", []))
                     if o is not None])
        cj = os.path.join(bio.candidates_dir(run_dir), r.frame + ".json")
        cl = []
        if tcfg.use_candidates and os.path.exists(cj):
            with open(cj) as f:
                for sh in json.load(f).get("bubbles", []):
                    o = make_obs(k, sh, "candidate", hw, s, tcfg.edge_px, score=sh.get("score", np.nan))
                    if o is not None:
                        cl.append(o)
        cands.append(cl)
    n_with_cands = sum(os.path.exists(os.path.join(bio.candidates_dir(run_dir), r.frame + ".json")) for r in lst.itertuples())
    if verbose:
        print(f"{K} frames, {sum(map(len, dets))} detections, {sum(map(len, cands))} low-score candidates "
              f"({n_with_cands} frames have candidates)  [{time.time() - t0:.0f}s]")
    tracks = link(dets, cands if tcfg.use_candidates else None, tcfg, s)
    n_interp = fill_gaps(tracks, metas[0][2], s, tcfg)
    n_join = join_broken(tracks, K, s, tcfg)
    ev = classify_events(tracks, K, s, tcfg)
    if verbose:
        n_resc = sum(o.source == "rescued" for obs in tracks.values() for o in obs)
        print(f"linked into {len(tracks)} tracks ({n_join} re-joined), {n_resc} rescued, {n_interp} interpolated  "
              f"[{time.time() - t0:.0f}s]")
    # renumber tracks by first appearance, then position
    order = sorted(tracks, key=lambda t: (tracks[t][0].k, tracks[t][0].cy, tracks[t][0].cx))
    new_id = {t: i for i, t in enumerate(order)}
    tracks = {new_id[t]: tracks[t] for t in order}
    ev = {new_id[t]: {**v, "start_partner": new_id.get(v["start_partner"], -1),
                      "end_partner": new_id.get(v["end_partner"], -1)} for t, v in ev.items()}
    for t, obs in tracks.items():
        for o in obs:
            o.tid = t
    bubbles, frames = _write_tracked(run_dir, lst, metas, tracks, cfg, tcfg, verbose)
    dt = frame_interval_us
    tr = track_table(bubbles, ev, dt)
    events = event_table(tr, bubbles, dt)
    td = tracked_dir(run_dir)
    tr.to_csv(os.path.join(td, "tracks.csv"), index=False)
    events.to_csv(os.path.join(td, "events.csv"), index=False)
    with open(os.path.join(td, "track_settings.json"), "w") as f:
        json.dump(dict(tracked=time.strftime("%Y-%m-%d %H:%M:%S"), include_unreviewed=include_unreviewed,
                       frame_interval_us=frame_interval_us, track_config=asdict(tcfg), config=asdict(cfg)),
                  f, indent=2, default=str)
    if verbose:
        print(f"wrote {td}/ (bubbles, frames, tracks, events)  [{time.time() - t0:.0f}s]")
    return bubbles, frames, tr, events


def _write_tracked(run_dir, lst, metas, tracks, cfg, tcfg, verbose):
    td = tracked_dir(run_dir)
    sd = bio.shapes_dir(td)
    os.makedirs(sd, exist_ok=True)
    per_k = [[] for _ in range(len(lst))]
    for t, obs in tracks.items():
        for o in obs:
            per_k[o.k].append(o)
    tables, frows = [], []
    for k, (r, a, hw) in enumerate(metas):
        obs = sorted(per_k[k], key=lambda o: o.tid)
        shapes = [dict(type=o.shape["type"], data=o.shape["data"], track_id=int(o.tid), source=o.source,
                       **({"fit_kind": o.shape["fit_kind"]} if "fit_kind" in o.shape else {})) for o in obs]
        img_rel = os.path.relpath(os.path.join(os.path.dirname(r.json_path), a["image"]), sd)
        with open(os.path.join(sd, r.frame + ".json"), "w") as f:
            json.dump(dict(image=img_rel, bubbles=shapes, rois=[], reviewed=bool(r.reviewed),
                           frame_path=a.get("frame_path", ""), model=a.get("model", ""), tracked=True), f)
        df = bio.measure_shapes(shapes, hw, cfg, frame=r.frame)
        if len(df):
            df.insert(1, "train", r.train)
            df.insert(2, "frame_idx", r.frame_idx)
            df.insert(3, "reviewed", r.reviewed)
            df.insert(4, "track_id", [shapes[i]["track_id"] for i in df["bubble_id"]])
            df.insert(5, "source", [shapes[i]["source"] for i in df["bubble_id"]])
            df.insert(6, "score", [obs[i].score for i in df["bubble_id"]])
            tables.append(df)
        src = pd.Series([o.source for o in obs], dtype=object)
        frows.append(dict(frame=r.frame, train=r.train, frame_idx=r.frame_idx, reviewed=r.reviewed,
                          n_bubbles=len(df), n_in_analysis=int(df["in_analysis"].sum()) if len(df) else 0,
                          n_detected=int((src == "detected").sum()), n_rescued=int((src == "rescued").sum()),
                          n_interpolated=int((src == "interpolated").sum()),
                          image_height_px=hw[0], image_width_px=hw[1], model=r.model))
    bubbles = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    frames = pd.DataFrame(frows)
    bubbles.to_csv(os.path.join(td, "bubbles.csv"), index=False)
    frames.to_csv(os.path.join(td, "frames.csv"), index=False)
    return bubbles, frames


def _slope(t, y):
    ok = np.isfinite(y)
    if ok.sum() < 3 or np.ptp(t[ok]) == 0:
        return np.nan
    return float(np.polyfit(t[ok], y[ok], 1)[0])


def track_table(bubbles: pd.DataFrame, ev: dict, dt: float, um_per_px: float = 3.2) -> pd.DataFrame:
    """One row per track. growth_um_per_us: slope of r_eq_um over time (linear fit); speed from fits of x, y."""
    rows = []
    for t, b in bubbles.sort_values("frame_idx").groupby("track_id"):
        tt = b["frame_idx"].to_numpy() * dt
        x, y = b["x"].to_numpy() * um_per_px, b["y"].to_numpy() * um_per_px
        src = b["source"]
        vx, vy = _slope(tt, x), _slope(tt, y)
        rows.append(dict(track_id=int(t), first_frame_idx=int(b["frame_idx"].iloc[0]),
                         last_frame_idx=int(b["frame_idx"].iloc[-1]), n_frames=len(b),
                         n_detected=int((src == "detected").sum()), n_rescued=int((src == "rescued").sum()),
                         n_interpolated=int((src == "interpolated").sum()),
                         t_first=tt[0], t_last=tt[-1], duration_us=tt[-1] - tt[0] + dt,
                         r_first_um=b["r_eq_um"].iloc[0], r_last_um=b["r_eq_um"].iloc[-1],
                         r_mean_um=b["r_eq_um"].mean(), r_max_um=b["r_eq_um"].max(),
                         growth_um_per_us=_slope(tt, b["r_eq_um"].to_numpy()),
                         speed_um_per_us=np.hypot(vx, vy) if np.isfinite(vx) else np.nan,
                         net_disp_um=float(np.hypot(x[-1] - x[0], y[-1] - y[0])),
                         path_um=float(np.hypot(np.diff(x), np.diff(y)).sum()),
                         touches_edge=bool(b["edge_truncated"].any()),
                         x_first=b["x"].iloc[0], y_first=b["y"].iloc[0], x_last=b["x"].iloc[-1], y_last=b["y"].iloc[-1],
                         **ev.get(int(t), {})))
    return pd.DataFrame(rows)


def event_table(tracks: pd.DataFrame, bubbles: pd.DataFrame, dt: float, um_per_px: float = 3.2) -> pd.DataFrame:
    """One row per start or end event (not for start/end of the train). frame_idx / t: the first frame in which the
    change is visible (first frame of a new bubble, first frame without an ending one)."""
    rows = []
    for r in tracks.itertuples():
        if r.end_event and r.end_event != "end_of_train":
            rows.append(dict(kind="end", event=r.end_event, track_id=r.track_id, partner=r.end_partner,
                             frame_idx=r.last_frame_idx + 1, t=(r.last_frame_idx + 1) * dt,
                             x_um=r.x_last * um_per_px, y_um=r.y_last * um_per_px, r_um=r.r_last_um,
                             volume_ratio=r.end_volume_ratio, track_frames=r.n_frames))
        if r.start_event and r.start_event != "start_of_train":
            rows.append(dict(kind="start", event=r.start_event, track_id=r.track_id, partner=r.start_partner,
                             frame_idx=r.first_frame_idx, t=r.first_frame_idx * dt,
                             x_um=r.x_first * um_per_px, y_um=r.y_first * um_per_px, r_um=r.r_first_um,
                             volume_ratio=r.start_volume_ratio, track_frames=r.n_frames))
    cols = ["kind", "event", "track_id", "partner", "frame_idx", "t", "x_um", "y_um", "r_um", "volume_ratio", "track_frames"]
    return pd.DataFrame(rows, columns=cols).sort_values(["frame_idx", "track_id"]).reset_index(drop=True)


def load_tracking(run_dir: str):
    """(bubbles, frames, tracks, events) of <run_dir>/tracked/."""
    td = tracked_dir(run_dir)
    return tuple(pd.read_csv(os.path.join(td, n + ".csv")) for n in ("bubbles", "frames", "tracks", "events"))


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
EVENT_STYLE = {          # event -> (label, colour)
    "merged": ("merged into another bubble", C[1]),
    "dissolved": ("dissolved (shrank, then vanished)", C[0]),
    "vanished": ("vanished (unexplained)", INK2),
    "appeared": ("appeared (new bubble)", C[2]),
    "emerged": ("emerged from an overlapping bubble", C[3]),
}


def track_color(tid):
    """A stable, distinct colour per track id (golden-ratio hue steps)."""
    import colorsys
    h = (tid * 0.6180339887) % 1.0
    return colorsys.hsv_to_rgb(h, 0.55 + 0.35 * ((tid * 0.37) % 1.0), 0.95)


def plot_tracking_qc(frames, time_label, smooth=1, ax=None):
    """Bubbles per frame as detected by the model vs after tracking, and how many were bridged per frame."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.2)) if ax is None else (None, ax)
    f = frames.sort_values("t")
    m = f.assign(t=f["t"])
    ends = [_series(axs[0], f["t"], f["n_detected"], C[1], many=True, smooth=smooth)[-1],
            _series(axs[0], f["t"], f["n_bubbles"], C[0], many=True, smooth=smooth)[-1]]
    axs[0].set_ylim(0, f["n_bubbles"].max() * 1.12)
    _end_labels(axs[0], _label_x(m), ends, ["detected by the model", "after tracking"])
    _room_right(axs[0], m, 0.28)
    axs[0].set_ylabel("bubbles per frame")
    _xaxis(axs[0], m, time_label)
    axs[0].set_title("Bubble count before / after tracking")
    ends = [_series(axs[1], f["t"], f["n_rescued"], C[2], many=True, smooth=smooth)[-1],
            _series(axs[1], f["t"], f["n_interpolated"], C[3], many=True, smooth=smooth)[-1]]
    axs[1].set_ylim(0, max(f["n_rescued"].max(), f["n_interpolated"].max(), 1) * 1.15)
    _end_labels(axs[1], _label_x(m), ends, ["rescued (low-score detection)", "interpolated (morphed outline)"])
    _room_right(axs[1], m, 0.4)
    axs[1].set_ylabel("bubbles per frame")
    _xaxis(axs[1], m, time_label)
    axs[1].set_title("Missed bubbles bridged by tracking")
    if fig is not None:
        fig.tight_layout()
    return fig


def plot_lifetimes(tracks, dt, ax=None):
    """How long bubbles are followed. Tracks cut by the start / end of the train are shown separately:
    their true lifetime is longer."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(7.5, 4.2))[1]
    t = tracks[tracks["end_event"] != "transient"]
    cut = (t["start_event"] == "start_of_train") | (t["end_event"] == "end_of_train")
    d = t["duration_us"]
    bins = np.geomspace(max(d.min(), dt) * 0.95, d.max() * 1.05, 25)
    ax.hist([d[~cut], d[cut]], bins=bins, stacked=True, color=[C[0], "#b7c9e2"], edgecolor="white", lw=0.5,
            label=[f"start and end seen ({(~cut).sum()})", f"cut by the train start / end ({cut.sum()})"])
    ax.set_xscale("log")
    ax.set_xlabel("time followed [µs]")
    ax.set_ylabel("bubbles")
    ax.legend(fontsize=8, loc="upper left")
    n_tr = (tracks["end_event"] == "transient").sum()
    ax.set_title(f"Bubble lifetimes ({n_tr} transient tracks not shown)")
    return ax


def plot_trajectories(bubbles, tracks, img, events=None, um_per_px=3.2, min_frames=10, cmap="RdYlGn_r"):
    """Left: path of every bubble followed for >= min_frames frames, coloured by its mean radius.
    Right: where bubbles merged, dissolved, vanished and appeared."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm
    H, W = img.shape[0] * um_per_px, img.shape[1] * um_per_px
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.2))
    long = tracks[tracks["n_frames"] >= min_frames].set_index("track_id")
    norm = LogNorm(vmin=max(long["r_mean_um"].min(), 0.5), vmax=long["r_mean_um"].max()) if len(long) else None
    b = bubbles[bubbles["track_id"].isin(long.index)].sort_values("frame_idx")
    segs, cols = [], []
    for tid, g in b.groupby("track_id"):
        segs.append(np.c_[g["x"] * um_per_px, g["y"] * um_per_px])
        cols.append(plt.get_cmap(cmap)(norm(long.loc[tid, "r_mean_um"])))
    for ax in axs:
        ax.imshow(img, cmap="gray", extent=[0, W, H, 0], alpha=0.45)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal"); ax.grid(False)
        ax.set_xlabel("x [µm]"); ax.set_ylabel("y [µm]")
    if segs:
        axs[0].add_collection(LineCollection(segs, colors=cols, linewidths=1.4))
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[0], shrink=0.8, label="mean radius [µm]")
    axs[0].set_title(f"Paths of {len(segs)} bubbles followed ≥ {min_frames} frames")
    if events is not None and len(events):
        for ev, (lab, col) in EVENT_STYLE.items():
            e = events[events["event"] == ev]
            if len(e):
                axs[1].scatter(e["x_um"], e["y_um"], s=12 + 2 * e["r_um"], color=col, alpha=0.8, lw=0.4,
                               edgecolor="white", label=f"{lab} ({len(e)})")
        axs[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    axs[1].set_title("Where bubbles merged, dissolved and appeared (marker size ~ radius)")
    fig.tight_layout()
    return fig


def plot_event_rates(events, frames, time_label, dt, bin_frames=20, tcfg: Optional[TrackConfig] = None):
    """Left: events per µs over time (binned). Right: volume balance of the mergers (1 = volume conserved)."""
    import matplotlib.pyplot as plt
    tcfg = tcfg or TrackConfig()
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.2), gridspec_kw=dict(width_ratios=[1.7, 1]))
    f0, f1 = int(frames["frame_idx"].min()), int(frames["frame_idx"].max())
    edges = np.arange(f0, f1 + bin_frames + 1, bin_frames)
    centres = (edges[:-1] + np.minimum(edges[1:], f1 + 1)) / 2 * dt
    width_us = (np.minimum(edges[1:], f1 + 1) - edges[:-1]) * dt
    series = [("merged", ["merged"]), ("dissolved", ["dissolved"]), ("vanished", ["vanished"]),
              ("appeared", ["appeared", "emerged"])]
    ends, labels = [], []
    m = pd.DataFrame(dict(t=centres))
    for name, evs in series:
        e = events[events["event"].isin(evs)]
        if not len(e):
            continue
        rate = np.histogram(e["frame_idx"], bins=edges)[0] / width_us
        lab = "appeared / emerged" if name == "appeared" else EVENT_STYLE[name][0].split(" (")[0]
        ends.append(_series(axs[0], centres, rate, EVENT_STYLE[name][1], many=len(centres) > 15, ms=5)[-1])
        labels.append(lab)
    if ends:
        _end_labels(axs[0], _label_x(m), ends, labels)
        _room_right(axs[0], m, 0.3)
    axs[0].set_ylim(bottom=0)
    axs[0].set_ylabel("events per µs")
    axs[0].set_xlabel(time_label)
    axs[0].set_title(f"Events over time ({bin_frames}-frame bins)")
    mv = events.loc[events["event"] == "merged", "volume_ratio"].dropna()
    if len(mv):
        axs[1].hist(mv.clip(upper=3), bins=np.linspace(0.5, 2.5, 21), color=C[1], edgecolor="white")
        axs[1].axvline(1, color=INK, lw=1, ls=":")
        axs[1].axvspan(*tcfg.volume_tol, color=C[1], alpha=0.08, lw=0)
    axs[1].set_xlabel("volume after / sum of volumes before")
    axs[1].set_ylabel("mergers")
    axs[1].set_title(f"Volume balance of {len(mv)} mergers")
    fig.tight_layout()
    return fig


def plot_growth(tracks, bubbles, dt, time_label, min_frames=20, n_long=40):
    """Left: growth rate dr/dt of each bubble vs its size (median per size bin). Ostwald ripening: small bubbles
    shrink, large ones grow; coalescence shows up as jumps instead (right).
    Right: radius over time of the bubbles followed longest; orange dots where another bubble merged into them."""
    import matplotlib.pyplot as plt
    from bubble_stats import _log_ticks, log_bins
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.6))
    t = tracks[(tracks["n_frames"] >= min_frames) & tracks["growth_um_per_us"].notna()]
    ax = axs[0]
    ax.set_xscale("log")
    ax.scatter(t["r_mean_um"], t["growth_um_per_us"], s=10, color=C[0], alpha=0.45, lw=0)
    ax.axhline(0, color=INK, lw=0.8, ls=":")
    if len(t) >= 5:
        bins = log_bins(t["r_mean_um"], n=8)
        idx = np.digitize(t["r_mean_um"], bins) - 1
        med = [(np.sqrt(bins[i] * bins[i + 1]), t["growth_um_per_us"][idx == i].median())
               for i in range(len(bins) - 1) if (idx == i).sum() >= 3]
        if med:
            ax.plot(*np.array(med).T, "-o", color=C[1], ms=5, mec="white", label="median per size bin")
            ax.legend(fontsize=8, loc="upper left")
        _log_ticks(ax, [v for v in (2, 5, 10, 20, 50, 100, 200) if bins[0] <= v <= bins[-1]])
    lim = np.nanpercentile(np.abs(t["growth_um_per_us"]), 98) * 1.3 if len(t) else 1
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("mean radius [µm]")
    ax.set_ylabel("growth rate dr/dt [µm/µs]")
    ax.set_title(f"Growth of individual bubbles (followed ≥ {min_frames} frames: {len(t)})")
    ax = axs[1]
    long = tracks[tracks["end_event"] != "transient"].nlargest(n_long, ["n_frames", "r_mean_um"])
    cols = plt.get_cmap("Blues")(np.linspace(0.35, 1.0, 256))
    rmax = long["r_mean_um"].max() if len(long) else 1
    rmin = long["r_mean_um"].min() if len(long) else 0
    b = bubbles[bubbles["track_id"].isin(long["track_id"])].sort_values("frame_idx")
    for tid, g in b.groupby("track_id"):
        c = cols[int(255 * (np.log(g["r_eq_um"].mean()) - np.log(rmin)) / max(np.log(rmax) - np.log(rmin), 1e-9))]
        ax.plot(g["frame_idx"] * dt, g["r_eq_um"], "-", color=c, lw=1.0)
    absorbed = tracks[(tracks["end_event"] == "merged") & tracks["end_partner"].isin(long["track_id"])]
    if len(absorbed):
        key = b.set_index(["track_id", "frame_idx"])["r_eq_um"]
        pts = [(f * dt, key.get((p, f))) for p, f in zip(absorbed["end_partner"], absorbed["last_frame_idx"] + 1)]
        pts = [p for p in pts if p[1] is not None and np.isfinite(p[1])]
        if pts:
            ax.plot(*np.array(pts).T, "o", color=C[1], ms=5, mec="white", mew=0.8, zorder=3,
                    label="another bubble merged into it")
            ax.legend(fontsize=8, loc="upper left")
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    _log_ticks(ax, [v for v in (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500) if lo <= v <= hi], axis="y")
    ax.set_xlabel(time_label)
    ax.set_ylabel("equivalent radius [µm]")
    ax.set_title(f"Radius over time of the {len(long)} longest-followed bubbles")
    fig.tight_layout()
    return fig


def plot_track_overlay(img, shapes, b, ax=None, alpha=0.4, title=None):
    """Bubbles coloured by track id (same colour in every frame); interpolated outlines dashed, unfilled."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    ax = ax or plt.subplots(figsize=(10, 6.5))[1]
    ax.imshow(img, cmap="gray")
    b = b.sort_values("area_px2", ascending=False)               # large first so small stay visible
    for interp in (False, True):
        sel = b[(b["source"] == "interpolated") == interp]
        if not len(sel):
            continue
        polys = [br.shape_outline(shapes[i])[:, ::-1] for i in sel["bubble_id"]]
        cols = [track_color(t) for t in sel["track_id"]]
        ax.add_collection(PolyCollection(
            polys, facecolors=[(*c, 0.0 if interp else alpha) for c in cols], edgecolors=[(*c, 1.0) for c in cols],
            linewidths=1.0 if interp else 0.6, linestyles="--" if interp else "-"))
    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(img.shape[0] - 0.5, -0.5)
    ax.set_title(title or f"{len(b)} bubbles", fontsize=10)
    ax.axis("off")
    return ax


def make_track_gif(run_dir, bubbles, m, out_path, every=1, fps=10, width_in=8.0, dpi=90, alpha=0.4,
                   time_fmt=lambda r: f"frame {r['frame_idx']}"):
    """GIF of the tracked bubbles, each bubble keeping its colour through the train (run_dir: the tracked dir)."""
    import matplotlib.pyplot as plt
    from bubble_stats import save_gif

    def figures():
        for _, r in m.iloc[::max(1, int(every))].iterrows():
            img, shapes = bio.load_frame(run_dir, r["frame"])
            b = bubbles[bubbles["frame"] == r["frame"]]
            n_i = int((b["source"] == "interpolated").sum())
            h = width_in * img.shape[0] / img.shape[1]
            fig, ax = plt.subplots(figsize=(width_in, h + 0.5))
            plot_track_overlay(img, shapes, b, ax=ax, alpha=alpha,
                               title=f"{time_fmt(r)}  ·  {len(b)} bubbles ({n_i} interpolated, dashed)")
            yield fig

    return save_gif(figures(), out_path, fps=fps, dpi=dpi)
