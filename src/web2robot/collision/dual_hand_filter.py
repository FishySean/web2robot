"""Post-process arm trajectories to remove *hand-vs-hand* interpenetration (B4).

Companion to ArmTorsoFilter (arm-vs-torso).  Uses the sphere-set hand proxies
in models.capsule_collision.HandSphereModel.

Unlike arm-through-torso — which is never legitimate — two hands *touching* is
usually intended (bimanual grasp: both hands on a jar, object held in front of
the chest).  Measurement on the official clips confirms this: every hand-hand
overlap observed (<=2.5 cm) coincides with a real two-handed action.  So this
filter is deliberately EVEN MORE conservative than the torso one:

  * it only fires when the two hands are clearly *passing through* each other
    (min sphere signed distance below -enter_thresh, default 4 cm), never on
    the shallow contact of a normal grasp;
  * when it does fire it pushes *both* arms symmetrically apart along the
    hand-to-hand axis while holding each hand near its human-derived target,
    so the grasp geometry is preserved as much as possible;
  * finger joints are never modified — the hands ride with the wrists.

The same penetration-guarded temporal smoothing as ArmTorsoFilter is applied.

Where the numbers live
----------------------
The constructor's defaults are read from `configs/robots/m7.yaml`
(`collision.dual_hand.defaults`) — one home for the parameter set, and the yaml
records that the conservative 4 cm threshold is a deliberate choice rather than a
calibrated one (there is no ground truth for "hands interpenetrating": B4's check
was on in-hand self-poking, which does not occur).  They remain real defaults, so
`inspect.signature(...)` still sees 0.04 as `tests/test_module_boundaries.py` pins.
"""

from __future__ import annotations

import numpy as np
import mujoco

from web2robot.robots.params import robot_params as _robot_params, values as _values

from .capsule_collision import HandSphereModel

_D = _values(_robot_params("m7")["collision"]["dual_hand"]["defaults"])


class DualHandFilter:
    def __init__(
        self,
        robot_cfg:    dict,
        enter_thresh: float = _D["enter_thresh"],   # only correct when hands overlap deeper than this [m]
        w_pen:        float = _D["w_pen"],   # push-apart weight (per metre of overlap)
        w_ee:         float = _D["w_ee"],    # hold each hand-frame position [per m^2]
        w_prox:       float = _D["w_prox"],  # stay near original joints [per rad^2]
        w_temp:       float = _D["w_temp"],  # stay near previous corrected frame
        max_iter:     int   = _D["max_iter"],
        lr:           float = _D["lr"],
        fd_eps:       float = _D["fd_eps"],
        smooth_sigma: float = _D["smooth_sigma"],
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
        self.smooth_sigma = smooth_sigma
        self.verbose = verbose

        self._env = robot_cfg["env_cls"](
            mjcf_path=robot_cfg["scene_path"],
            start_config=robot_cfg["start_config"],
        )
        self._env.reset()
        self._model = self._env.model
        self._data  = self._env.data
        self._hm = HandSphereModel(self._model)

        self._hand_bid = {
            s: mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_hand_frame")
            for s in ("left", "right")
        }
        if verbose:
            print(f"[DualHandFilter] enter_thresh={enter_thresh:.3f}m "
                  f"w_pen={w_pen} w_ee={w_ee}")

    # ── kinematics helpers ──────────────────────────────────────────────────

    def _forward_both(self, qL, qR):
        self._env.set_arm_joints("left",  qL.astype(np.float64))
        self._env.set_arm_joints("right", qR.astype(np.float64))
        mujoco.mj_forward(self._model, self._data)

    def _hh(self) -> float:
        d, _ = self._hm.hand_hand_min(self._data)
        return d

    def _hand_pos(self, side):
        return self._data.xpos[self._hand_bid[side]].copy()

    def _set_fingers(self, side, vals, finger_jnames):
        names = [f"{side}_{jn}" for jn in finger_jnames]
        self._env.set_finger_joints(np.asarray(vals, dtype=np.float64), names)

    # ── per-frame optimisation over BOTH arms (14 dof) ──────────────────────

    def _loss(self, q, qL0, qR0, qLp, qRp, tL, tR):
        qL, qR = q[:7], q[7:]
        self._forward_both(qL, qR)
        pen  = self.w_pen * max(0.0, -self._hh())
        ee   = self.w_ee * (float(np.sum((self._hand_pos("left")  - tL) ** 2)) +
                            float(np.sum((self._hand_pos("right") - tR) ** 2)))
        prox = self.w_prox * (float(np.dot(qL - qL0, qL - qL0)) +
                             float(np.dot(qR - qR0, qR - qR0)))
        temp = self.w_temp * (float(np.dot(qL - qLp, qL - qLp)) +
                             float(np.dot(qR - qRp, qR - qRp)))
        return pen + ee + prox + temp

    def _gradient(self, q, *a):
        eps = self.fd_eps
        f0 = self._loss(q, *a)
        g = np.empty_like(q)
        for i in range(len(q)):
            q[i] += eps
            g[i] = (self._loss(q, *a) - f0) / eps
            q[i] -= eps
        return g

    def _optimize(self, qL0, qR0, qLp, qRp, tL, tR):
        q = np.concatenate([qL0, qR0]).astype(np.float64)
        a = (qL0, qR0, qLp, qRp, tL, tR)
        best_q, best_loss = q.copy(), float("inf")
        for _ in range(self.max_iter):
            self._forward_both(q[:7], q[7:])
            if self._hh() >= 0.0:
                return q[:7].copy(), q[7:].copy(), True
            loss = self._loss(q, *a)
            if loss < best_loss:
                best_loss, best_q = loss, q.copy()
            q = q - self.lr * self._gradient(q, *a)
        self._forward_both(best_q[:7], best_q[7:])
        return best_q[:7].copy(), best_q[7:].copy(), self._hh() >= 0.0

    # ── public API ──────────────────────────────────────────────────────────

    def process(self, q_left, q_right,
                q_left_fingers=None, q_right_fingers=None, finger_jnames=None):
        T = q_left.shape[0]
        qL = q_left.copy(); qR = q_right.copy()
        have_fingers = finger_jnames is not None
        corrected = np.zeros(T, bool)
        bad = fixed = 0

        for t in range(T):
            if have_fingers:
                self._set_fingers("left",  q_left_fingers[t],  finger_jnames)
                self._set_fingers("right", q_right_fingers[t], finger_jnames)
            self._forward_both(qL[t], qR[t])
            if self._hh() >= -self.enter_thresh:      # conservative gate
                continue
            bad += 1
            corrected[t] = True
            tL = self._hand_pos("left")
            tR = self._hand_pos("right")
            qLp = qL[t - 1] if t > 0 else qL[t]
            qRp = qR[t - 1] if t > 0 else qR[t]
            nL, nR, ok = self._optimize(qL[t], qR[t], qLp, qRp, tL, tR)
            qL[t], qR[t] = nL, nR
            if ok:
                fixed += 1

        smoothed = 0
        if self.smooth_sigma > 0 and corrected.any():
            smoothed = self._smooth(qL, qR, q_left_fingers, q_right_fingers,
                                    finger_jnames if have_fingers else None,
                                    corrected)

        if self.verbose:
            if bad == 0:
                print("[DualHandFilter] no hand-hand penetration")
            else:
                print(f"[DualHandFilter] fixed {fixed}/{bad} "
                      f"(remaining {bad - fixed}); smoothed {smoothed} frames")
        return qL, qR

    def _smooth(self, qL, qR, fL, fR, finger_jnames, corrected):
        from scipy.ndimage import gaussian_filter1d
        qL0, qR0 = qL.copy(), qR.copy()
        qLs = gaussian_filter1d(qL0, self.smooth_sigma, axis=0, mode="nearest")
        qRs = gaussian_filter1d(qR0, self.smooth_sigma, axis=0, mode="nearest")
        radius = int(np.ceil(2 * self.smooth_sigma))
        region = np.convolve(corrected.astype(int),
                             np.ones(2 * radius + 1, int), mode="same") > 0
        n = 0
        for t in np.where(region)[0]:
            if finger_jnames is not None:
                self._set_fingers("left",  fL[t], finger_jnames)
                self._set_fingers("right", fR[t], finger_jnames)
            alpha = 1.0
            self._forward_both(qLs[t], qRs[t])
            if self._hh() < -self.enter_thresh:
                lo, hi = 0.0, 1.0
                for _ in range(6):
                    alpha = 0.5 * (lo + hi)
                    self._forward_both((1 - alpha) * qL0[t] + alpha * qLs[t],
                                       (1 - alpha) * qR0[t] + alpha * qRs[t])
                    if self._hh() >= -self.enter_thresh:
                        lo = alpha
                    else:
                        hi = alpha
                alpha = lo
            qL[t] = (1 - alpha) * qL0[t] + alpha * qLs[t]
            qR[t] = (1 - alpha) * qR0[t] + alpha * qRs[t]
            if alpha > 0.0 and not (np.allclose(qL[t], qL0[t]) and np.allclose(qR[t], qR0[t])):
                n += 1
        return n
