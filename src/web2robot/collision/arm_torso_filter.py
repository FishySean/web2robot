"""Post-process an arm trajectory to remove arm-vs-torso self-collision (B1).

Companion to models.collision.CollisionFilter (which handles *cross-arm*
contacts via MuJoCo geoms).  m7.xml DOES have 98 collision-enabled mesh geoms
(the disabled ones are in the arms-only m7_mjx.xml training model) — but the
upstream cross-arm filter still cannot see arm-vs-torso, because its geom sets
exclude the shared waist bodies by construction and its chain walk excludes the
whole hand.  See collision/__init__.py for the measurements.  So arm-through-body
needs a separate proxy-based pass — that is this module.

Detection uses the analytic capsule/box proxies in models.capsule_collision
(no model surgery).  Correction is per-frame, per-side finite-difference
gradient descent, deliberately CONSERVATIVE:

  * a frame is only touched when a bone is clearly *inside* the torso box
    (signed distance below -enter_thresh); shallow grazing is left alone;
  * the hand-frame world position is held via a strong fidelity term, so the
    correction is resolved through the arm's redundancy (elbow swivel) and the
    retargeted wrist target — which comes from the human hand — is preserved as
    much as possible.  We push the arm just out of the body, not further.

Only the offending side's 7 joints move; the other arm is untouched.
"""

from __future__ import annotations

import numpy as np
import mujoco

from .capsule_collision import M7CapsuleModel


class ArmTorsoFilter:
    def __init__(
        self,
        robot_cfg:   dict,
        enter_thresh: float = 0.04,   # only correct bones deeper than this [m]
        w_pen:        float = 20.0,   # push-out weight (per metre of penetration)
        w_ee:         float = 60.0,   # hold hand-frame position [per m^2]
        w_prox:       float = 1.0,    # stay near original joints [per rad^2]
        w_temp:       float = 0.5,    # stay near previous corrected frame
        max_iter:     int   = 60,
        lr:           float = 0.03,
        fd_eps:       float = 1e-3,
        include_fingers: bool = True,   # fold fingertip-in-torso into detection
        smooth_sigma: float = 2.0,      # temporal smoothing of corrected frames [frames]
        verbose:      bool  = True,
    ):
        self.enter_thresh = enter_thresh
        self.w_pen  = w_pen
        self.w_ee   = w_ee
        self.w_prox = w_prox
        self.w_temp = w_temp
        self.max_iter = max_iter
        self.lr = lr
        self.fd_eps = fd_eps
        self._include_fingers = include_fingers
        self.smooth_sigma = smooth_sigma
        self.verbose = verbose

        self._env = robot_cfg["env_cls"](
            mjcf_path=robot_cfg["scene_path"],
            start_config=robot_cfg["start_config"],
        )
        self._env.reset()
        self._model = self._env.model
        self._data  = self._env.data
        self._cm = M7CapsuleModel(self._model)

        self._hand_bid = {
            "left":  mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_frame"),
            "right": mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_frame"),
        }
        if verbose:
            print(f"[ArmTorsoFilter] enter_thresh={enter_thresh:.3f}m "
                  f"w_pen={w_pen} w_ee={w_ee} w_prox={w_prox}")

    # ── kinematics helpers ──────────────────────────────────────────────────

    def _forward(self, side: str, q: np.ndarray) -> None:
        self._env.set_arm_joints(side, q.astype(np.float64))
        mujoco.mj_forward(self._model, self._data)

    def _sdf(self, side: str) -> float:
        return self._cm.arm_torso_penetrations(
            self._data, include_fingers=self._include_fingers)[side]

    def _hand_pos(self, side: str) -> np.ndarray:
        return self._data.xpos[self._hand_bid[side]].copy()

    # ── per-side per-frame optimisation ─────────────────────────────────────

    def _loss(self, side, q, q_ref, ee_target) -> float:
        self._forward(side, q)
        sdf = self._sdf(side)
        pen = self.w_pen * max(0.0, -sdf)              # drive bone out of the box
        ee  = self.w_ee * float(np.sum((self._hand_pos(side) - ee_target) ** 2))
        prox = self.w_prox * float(np.dot(q - q_ref[0], q - q_ref[0]))
        temp = self.w_temp * float(np.dot(q - q_ref[1], q - q_ref[1]))
        return pen + ee + prox + temp

    def _gradient(self, side, q, q_ref, ee_target) -> np.ndarray:
        eps = self.fd_eps
        f0 = self._loss(side, q, q_ref, ee_target)
        g = np.empty_like(q)
        for i in range(len(q)):
            q[i] += eps
            g[i] = (self._loss(side, q, q_ref, ee_target) - f0) / eps
            q[i] -= eps
        return g

    def _optimize_side(self, side, q0, q_prev, ee_target):
        """Push one arm out of the torso; return (q, resolved)."""
        q_ref = (q0.copy(), q_prev.copy())   # (proximity anchor, temporal anchor)
        q = q0.copy()
        best_q, best_loss = q.copy(), float("inf")
        for _ in range(self.max_iter):
            self._forward(side, q)
            if self._sdf(side) >= 0.0:        # out of the body — done
                return q, True
            loss = self._loss(side, q, q_ref, ee_target)
            if loss < best_loss:
                best_loss, best_q = loss, q.copy()
            q = q - self.lr * self._gradient(side, q, q_ref, ee_target)
        self._forward(side, best_q)
        return best_q, self._sdf(side) >= 0.0

    def _set_fingers(self, side, vals, finger_jnames):
        """Write one side's finger qpos (via the env's name mapping) so fingertip
        proxies reflect the real hand pose during evaluation.  Finger joints are
        never optimised — they just ride with the wrist as the arm is corrected."""
        # env.set_finger_joints expects side-prefixed short names; retargeter
        # names come bare (e.g. "thumb_bend") -> prefix here.
        names = [f"{side}_{jn}" for jn in finger_jnames]
        self._env.set_finger_joints(np.asarray(vals, dtype=np.float64), names)

    # ── public API ──────────────────────────────────────────────────────────

    def process(self, q_left, q_right,
                q_left_fingers=None, q_right_fingers=None, finger_jnames=None):
        """Remove arm-vs-torso penetration; returns corrected copies.

        If per-frame finger trajectories are supplied, fingertip-in-torso is
        included in detection (a hand poking into the body then triggers an
        arm correction); the finger joints themselves are left untouched.
        """
        T = q_left.shape[0]
        out = {"left": q_left.copy(), "right": q_right.copy()}
        fingers = {"left": q_left_fingers, "right": q_right_fingers}
        have_fingers = finger_jnames is not None
        stats = {"left": [0, 0], "right": [0, 0]}   # [bad, fixed]
        corrected = {"left": np.zeros(T, bool), "right": np.zeros(T, bool)}

        for side in ("left", "right"):
            q = out[side]
            fj = fingers[side]
            for t in range(T):
                if fj is not None and have_fingers:
                    self._set_fingers(side, fj[t], finger_jnames)
                self._forward(side, q[t])
                sdf = self._sdf(side)
                if sdf >= -self.enter_thresh:     # conservative gate
                    continue
                stats[side][0] += 1
                corrected[side][t] = True
                # preserve the human-derived hand target for this frame
                ee_target = self._hand_pos(side)
                q_prev = q[t - 1] if t > 0 else q[t]
                q_new, resolved = self._optimize_side(side, q[t], q_prev, ee_target)
                q[t] = q_new
                if resolved:
                    stats[side][1] += 1

        # temporal smoothing of the corrected frames (kills the frame-to-frame
        # jitter that per-frame independent optimisation introduces), guarded so
        # it never re-introduces clear penetration.
        smoothed = {"left": 0, "right": 0}
        if self.smooth_sigma > 0:
            for side in ("left", "right"):
                if corrected[side].any():
                    smoothed[side] = self._smooth_side(
                        side, out[side], fingers[side],
                        finger_jnames if have_fingers else None,
                        corrected[side])

        if self.verbose:
            for side in ("left", "right"):
                bad, fixed = stats[side]
                if bad == 0:
                    print(f"[ArmTorsoFilter] {side}: no arm-torso penetration")
                else:
                    print(f"[ArmTorsoFilter] {side}: fixed {fixed}/{bad} "
                          f"(remaining {bad - fixed}); smoothed {smoothed[side]} frames")
        return out["left"], out["right"]

    def _smooth_side(self, side, q, fj, finger_jnames, corrected):
        """Gaussian-smooth the corrected region of one arm's joint trajectory.

        Only frames in/adjacent to a correction are touched (clean regions keep
        the retargeter's own motion).  Each smoothed frame is re-checked against
        the torso; if smoothing pushed it back past -enter_thresh, we blend it
        back toward the (un-smoothed) corrected pose just enough to clear — so
        smoothness is maximised without ever re-creating clear penetration.
        """
        from scipy.ndimage import gaussian_filter1d
        T = q.shape[0]
        q_corr = q.copy()
        q_sm = gaussian_filter1d(q_corr, self.smooth_sigma, axis=0, mode="nearest")
        radius = int(np.ceil(2 * self.smooth_sigma))
        region = np.convolve(corrected.astype(int),
                             np.ones(2 * radius + 1, int), mode="same") > 0
        n = 0
        for t in np.where(region)[0]:
            if fj is not None and finger_jnames is not None:
                self._set_fingers(side, fj[t], finger_jnames)
            # binary-search the blend that keeps sdf >= -enter_thresh
            alpha = 1.0                                   # 1 = fully smoothed
            self._forward(side, q_sm[t])
            if self._sdf(side) < -self.enter_thresh:
                lo, hi = 0.0, 1.0
                for _ in range(6):
                    alpha = 0.5 * (lo + hi)
                    self._forward(side, (1 - alpha) * q_corr[t] + alpha * q_sm[t])
                    if self._sdf(side) >= -self.enter_thresh:
                        lo = alpha
                    else:
                        hi = alpha
                alpha = lo
            q[t] = (1 - alpha) * q_corr[t] + alpha * q_sm[t]
            if alpha > 0.0 and not np.allclose(q[t], q_corr[t]):
                n += 1
        return n
