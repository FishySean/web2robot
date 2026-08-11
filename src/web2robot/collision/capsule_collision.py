"""Capsule/box collision proxies for M7 arm-vs-torso self-collision (B0).

Model-agnostic: we do NOT touch the MuJoCo model (we neither add nor enable any
geom; m7.xml's own 98 collision geoms are left exactly as they are and are not
read here).  Instead we attach lightweight analytic proxies to a few bodies and
read their world poses (data.xpos / data.xmat) after mj_forward:

  * each long arm bone (upper arm, forearm) -> a capsule (segment + radius),
    the segment being the vector from the link origin to its child link origin
    (read straight from model.body_pos, so it is exact for the kinematics);
  * the torso -> an oriented box (OBB) on waist_pitch_link.  A box (not a
    capsule) is used on purpose: the trunk is wide (y) and shallow (x), so an
    isotropic capsule would flag arms that hang naturally at the sides.  The
    box's anisotropy lets a forearm sit beside the torso without a false hit,
    and only fires when a bone actually crosses into the trunk volume.

Signed distance is arm-capsule vs torso-box: negative == penetration.  The
torso box is deliberately CONSERVATIVE (shrunk half-extents) so only clear
"arm through the body" cases go negative, per the agreed conservative policy.
"""

import mujoco
import numpy as np


# ---- geometry helpers -------------------------------------------------------

def _point_box_sdf(p_local, half):
    """Signed distance from a point (in box-local frame) to an axis-aligned box.

    Negative inside, positive outside.  Standard smooth-ish box SDF.
    """
    d = np.abs(p_local) - half
    outside = np.linalg.norm(np.maximum(d, 0.0))
    inside = min(0.0, float(np.max(d)))  # 0 if outside, negative if inside
    return outside + inside


def _capsule_box_sdf(a_world, b_world, radius, box_center, box_R, box_half, n=10):
    """Signed distance between a capsule (segment a-b, radius) and an OBB.

    Sample n points along the segment, take min point-box SDF, subtract radius.
    Sampling (vs exact seg-OBB) is smooth enough for finite-diff descent and
    keeps the code simple; 10 samples on a ~0.3 m bone is plenty.
    """
    Rt = box_R.T
    best = np.inf
    for t in np.linspace(0.0, 1.0, n):
        p = a_world + t * (b_world - a_world)
        p_local = Rt @ (p - box_center)
        best = min(best, _point_box_sdf(p_local, box_half))
    return best - radius


# ---- proxy model ------------------------------------------------------------

class M7CapsuleModel:
    """Holds proxy definitions and evaluates them against a MuJoCo mjData.

    Usage:
        cm = M7CapsuleModel(model)
        # ... set arm qpos on data, then mujoco.mj_forward(model, data) ...
        pens = cm.arm_torso_penetrations(data)   # dict side -> min signed dist
    """

    # torso box: on waist_pitch_link, from the trunk mesh AABB
    # (pos=[-0.003,0,0.24], size=[0.139,0.17,0.239]).  Shrunk to be conservative
    # so arms grazing the body surface are not flagged, only real pass-through.
    TORSO_BODY = "waist_pitch_link"
    TORSO_CENTER = np.array([-0.003, 0.0, 0.24], dtype=np.float64)
    TORSO_HALF = np.array([0.105, 0.135, 0.215], dtype=np.float64)  # shrunk from .139/.17/.239

    # arm bones: (parent link, child link giving the segment vector, radius).
    # radius from the link geom cross-section half-extent (upper ~0.05, fore ~0.045).
    BONES = {
        "left": [
            ("left_arm_yaw_link", "left_elbow_pitch_link", 0.050),   # upper arm
            ("left_elbow_pitch_link", "left_elbow_yaw_link", 0.045),  # forearm
        ],
        "right": [
            ("right_arm_yaw_link", "right_elbow_pitch_link", 0.050),
            ("right_elbow_pitch_link", "right_elbow_yaw_link", 0.045),
        ],
    }

    # fingertip proxies (distal link of each finger).  Used only for detection —
    # a hand poking into the torso should trigger an *arm* correction that moves
    # the whole hand out; finger joints themselves are never modified.  The tips
    # ride rigidly with the wrist, so their world positions reflect the actual
    # (curled/extended) hand pose as long as the finger qpos is set before eval.
    FINGERTIPS = {
        "left": ["left_hand_thumb_rota_link2", "left_hand_index_rota_link2",
                 "left_hand_mid_link2", "left_hand_ring_link2",
                 "left_hand_pinky_link2"],
        "right": ["right_hand_thumb_rota_link2", "right_hand_index_rota_link2",
                  "right_hand_mid_link2", "right_hand_ring_link2",
                  "right_hand_pinky_link2"],
    }
    TIP_RADIUS = 0.012

    def __init__(self, model):
        self.model = model
        bid = lambda n: mujoco.mj_id2name and mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, n)
        self.torso_bid = bid(self.TORSO_BODY)
        # resolve bones -> (parent_bid, local segment vector, radius)
        self.bones = {}
        for side, specs in self.BONES.items():
            out = []
            for parent, child, r in specs:
                pbid = bid(parent)
                cbid = bid(child)
                seg = model.body_pos[cbid].astype(np.float64)  # child origin in parent frame
                out.append((pbid, seg, float(r)))
            self.bones[side] = out
        # resolve fingertip body ids
        self.tips = {side: [bid(n) for n in names]
                     for side, names in self.FINGERTIPS.items()}

    def _torso_frame(self, data):
        R = data.xmat[self.torso_bid].reshape(3, 3)
        center = data.xpos[self.torso_bid] + R @ self.TORSO_CENTER
        return center, R

    def arm_torso_penetrations(self, data, margin=0.0, include_fingers=False):
        """Return {side: min signed distance} of that arm's bones vs torso box.

        signed distance < 0  => penetration.  `margin` shifts the threshold
        (a positive margin demands clearance).  Reported value has margin
        subtracted so callers can just test < 0.  With include_fingers, the
        side's fingertip spheres are folded in so a hand poking into the torso
        also registers (requires the finger qpos to be set in `data`).
        """
        center, R = self._torso_frame(data)
        out = {}
        for side, bones in self.bones.items():
            worst = np.inf
            for pbid, seg, r in bones:
                Rp = data.xmat[pbid].reshape(3, 3)
                a = data.xpos[pbid]
                b = a + Rp @ seg
                d = _capsule_box_sdf(a, b, r, center, R, self.TORSO_HALF)
                worst = min(worst, d)
            if include_fingers:
                for tbid in self.tips[side]:
                    p_local = R.T @ (data.xpos[tbid] - center)
                    worst = min(worst, _point_box_sdf(p_local, self.TORSO_HALF) - self.TIP_RADIUS)
            out[side] = worst - margin
        return out

    def all_bone_worlds(self, data):
        """Debug: world endpoints+radius of every bone capsule, per side."""
        out = {}
        for side, bones in self.bones.items():
            segs = []
            for pbid, seg, r in bones:
                Rp = data.xmat[pbid].reshape(3, 3)
                a = data.xpos[pbid].copy()
                b = a + Rp @ seg
                segs.append((a, b, r))
            out[side] = segs
        return out


class HandSphereModel:
    """Sphere-set proxies for each hand, for hand-vs-hand and intra-hand
    finger collision detection (model-agnostic; reads body world poses).

    Each hand is a small set of spheres: one palm sphere on *_hand_frame plus
    two spheres per finger (proximal link1 + distal link2).  Signed distance
    between two spheres is ||c_a - c_b|| - r_a - r_b (negative == overlap).

    Detection is deliberately CONSERVATIVE: adjacent fingers within a hand sit
    naturally side-by-side, so intra-hand queries exclude same-finger and
    neighbouring-finger pairs and only report spheres that are physically far
    apart on the hand yet overlapping in space (a real cross-over / self-poke).
    """

    # (body suffix, radius, finger_id) — finger_id groups spheres per digit so
    # same/adjacent-finger pairs can be excluded in intra-hand queries.
    #   0 thumb  1 index  2 mid  3 ring  4 pinky   (-1 = palm)
    PALM_RADIUS = 0.035
    LINK_RADIUS = 0.013
    _SPHERES = [
        ("hand_frame",        PALM_RADIUS, -1),
        ("hand_thumb_rota_link1", LINK_RADIUS, 0),
        ("hand_thumb_rota_link2", LINK_RADIUS, 0),
        ("hand_index_rota_link1", LINK_RADIUS, 1),
        ("hand_index_rota_link2", LINK_RADIUS, 1),
        ("hand_mid_link1",    LINK_RADIUS, 2),
        ("hand_mid_link2",    LINK_RADIUS, 2),
        ("hand_ring_link1",   LINK_RADIUS, 3),
        ("hand_ring_link2",   LINK_RADIUS, 3),
        ("hand_pinky_link1",  LINK_RADIUS, 4),
        ("hand_pinky_link2",  LINK_RADIUS, 4),
    ]

    def __init__(self, model):
        self.model = model
        bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        self.spheres = {}
        for side in ("left", "right"):
            out = []
            for suffix, r, fid in self._SPHERES:
                out.append((bid(f"{side}_{suffix}"), float(r), fid))
            self.spheres[side] = out

    def _centers(self, data, side):
        return [(data.xpos[b].copy(), r, fid) for b, r, fid in self.spheres[side]]

    def hand_hand_min(self, data):
        """Min signed distance between any left-sphere and any right-sphere.

        Returns (min_dist, (left_center, right_center)); negative == overlap.
        """
        L = self._centers(data, "left")
        R = self._centers(data, "right")
        best, pair = np.inf, (None, None)
        for cl, rl, _ in L:
            for cr, rr, _ in R:
                d = float(np.linalg.norm(cl - cr)) - rl - rr
                if d < best:
                    best, pair = d, (cl, cr)
        return best, pair

    def intra_hand_min(self, data, side, finger_gap=2):
        """Min signed distance between spheres of *non-neighbouring* fingers
        within one hand (|finger_id difference| >= finger_gap; palm excluded).

        finger_gap=2 skips same-finger and immediate neighbours (thumb-index,
        index-mid, ...), which are naturally close, so only a genuine cross-over
        (e.g. thumb vs ring/pinky, index vs pinky) can register.  Returns the
        min signed distance (negative == overlap).
        """
        S = [(c, r, fid) for c, r, fid in self._centers(data, side) if fid >= 0]
        best = np.inf
        for i in range(len(S)):
            ci, ri, fi = S[i]
            for j in range(i + 1, len(S)):
                cj, rj, fj = S[j]
                if abs(fi - fj) < finger_gap:
                    continue
                d = float(np.linalg.norm(ci - cj)) - ri - rj
                best = min(best, d)
        return best
