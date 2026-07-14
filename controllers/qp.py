"""Receding-horizon Quadratic Program (QP) lateral controller.

Approach:
  * Plant identified as a linear model lat[k] = A*lat[k-1] + B*steer[k] + C*roll[k]
    fit from ground-truth logged data (R^2 = 0.988).
  * The predicted lataccel over the horizon is linear in the steer vector, and the
    tracking + jerk + steer-rate + steer-reference costs are quadratic, so each step
    is a single precomputed closed-form least-squares solve (no iterative solver).
  * A model-inverting steady-state feed-forward anchors the steer; a small integral
    with back-calculation anti-windup removes steady-state lag without windup.
  * The steer->lataccel gain rises with |lataccel|, so both the feed-forward gain and
    the QP's prediction gain are scheduled on target magnitude (sched_gain / sched_qp).
  * A soft start blends feed-forward -> full QP over the first control steps to avoid
    overdriving the handoff transient at CONTROL_START_IDX.

~57 mean total_cost over 500 segments (pid baseline ~104).
"""
from . import BaseController
import numpy as np


# Linear plant model fit from ground-truth logged data over 500 segments
# (calibrate_dynamics.py, R^2 = 0.988):
#   lataccel[k] = A*lataccel[k-1] + B*steer[k] + C*roll[k]
A = 0.8857
B = 0.1806
C = 0.1189

HORIZON = 40          # planning steps (future_plan provides up to ~50)
DEL_T = 0.1
STEER_MIN, STEER_MAX = -2.0, 2.0

# Cost weights.
#   LAT  mirrors the competition's 50x tracking weight.
#   JERK penalizes predicted lataccel changes.
#   STEER_RATE penalizes changes in the commanded steer (smoothness).
#   STEER_REF is the key stabilizer: it pulls the steer toward the modest
#     steady-state feed-forward value u_ss. Without it the QP overdrives and
#     saturates the steer toward +/-2, pushing the ONNX plant out of its
#     training distribution where it oscillates. Anchoring to u_ss keeps the
#     commanded steer in-distribution (the same regime ff_pid uses) while the
#     QP still refines around it for tracking + jerk.
# NOTE on stability: the tracking term alone wants u ~ error/B, i.e. an effective
# proportional gain of 1/B ~ 6 — far above the stable range (ff_pid is stable at
# p=0.11). That high feedback gain causes a period-2 limit cycle. What sets the
# effective gain is the RATIO of tracking weight to the regularizers, so we keep
# tracking modest and let the steer-reference (open-loop feed-forward) dominate.
# Tuned via coordinate sweeps over 30 segments (tune_qp.py): total_cost ~42.5.
# The optimum is a broad flat basin — these sit mid-basin for robustness.
LAT_WEIGHT = 14.0
JERK_WEIGHT = 5.0
STEER_RATE_WEIGHT = 2.0
STEER_REF_WEIGHT = 12.0

# Integral gain. The QP tracking term is proportional-style and leaves a
# steady-state lag during sustained maneuvers (the slow plant never catches up).
# A small integral correction eliminates that offset. It is low-frequency, so it
# cannot trigger the high-frequency period-2 oscillation that high proportional
# feedback does.
I_GAIN = 0.12
I_CLIP = 2.0   # anti-windup clamp on the accumulated integral correction


class Controller(BaseController):
  """
  Receding-horizon Quadratic Program controller.

  Predicted lataccel over the horizon is linear in the steer vector u:
  lat = G @ u + f.  Tracking, jerk, and steer-rate costs are all quadratic in u,
  so the optimum is one closed-form least-squares solve.  The Hessian is constant
  (only the linear term changes), so we precompute the solver P once and each
  step is  u = P @ c.

  Only u[0] is applied; the next call re-solves with the true current lataccel,
  giving closed-loop feedback that corrects model mismatch.
  """
  def __init__(self, horizon=HORIZON, lat_weight=LAT_WEIGHT, jerk_weight=JERK_WEIGHT,
               steer_rate_weight=STEER_RATE_WEIGHT, steer_ref_weight=STEER_REF_WEIGHT,
               i_gain=I_GAIN, i_clip=I_CLIP, a=A, b=B, c=C, lat_filter=1.0,
               sched_gain=True, sched_qp=True, soft_steps=10, takeover_call=80):
    H = horizon
    self.H = H
    self.prev_steer = 0.0
    self.i_gain = i_gain
    self.i_clip = i_clip
    self.error_integral = 0.0

    # Soft start: update() is called every step from CONTEXT_LENGTH (20); control
    # takes over at CONTROL_START_IDX (100) -> the 80th call. At takeover the plant
    # switches from forced-perfect to a stochastic prediction whose first value
    # jumps; the QP would overdrive to erase that jump and overshoot ~10x. For the
    # first `soft_steps` control steps we blend from the smooth feed-forward
    # (alpha=0) to the full QP (alpha=1) so it doesn't chase the handoff noise.
    self.soft_steps = soft_steps
    self.takeover_call = takeover_call
    self.call_count = 0
    # EMA smoothing factor applied to current_lataccel before it enters the QP.
    # 1.0 = no filtering (raw measurement). Lower values denoise the feedback so
    # the controller chases plant sampling noise less -> lower jerk (at the risk
    # of a little lag). The integral term still uses the raw error for accuracy.
    self.lat_filter = lat_filter
    self.lat_filt = None

    # Steady-state inverse gains derived from the dynamics
    self.steer_ss_gain = b / (1 - a)
    self.roll_ss_gain = c / (1 - a)

    # The steer->lataccel gain is not constant: it rises steeply with |lataccel|
    # (measured B/(1-A) per regime: ~1.25 below 1, ~1.74 at 1-2, ~1.96 at 2-3).
    # The global gain (1.58) over-commands steer on aggressive targets, saturating
    # the actuator on maneuvers that are actually achievable. Schedule the
    # feed-forward gain on the target magnitude to fix this. (sched_gain toggles it.)
    # Keep the normal regime at the global gain (the QP's internal dynamics use
    # it, so FF and QP stay consistent) and only ramp up for aggressive targets,
    # where the global gain over-commands into saturation. Scheduling the low
    # regime down hurt the median by making FF and QP disagree.
    self.sched_gain = sched_gain
    self._gain_lat = np.array([0.0, 1.5, 2.0, 2.5, 5.0])
    self._gain_val = np.array([self.steer_ss_gain, self.steer_ss_gain,
                               1.74, 1.96, 1.96])

    # Roll/free-response matrices and difference operators are gain-independent.
    # Rc[i, j] = a^(i-j) * c   (roll -> lat impulse response)
    # Apow[i]  = a^(i+1)       (free response of initial lat)
    Rc = np.zeros((H, H))
    for i in range(H):
      for j in range(i + 1):
        Rc[i, j] = (a ** (i - j)) * c
    self.Rc = Rc
    self.Apow = np.array([a ** (i + 1) for i in range(H)])
    self.E = np.eye(H) - np.eye(H, k=-1)   # jerk: first-diff on predicted lat
    self.D = np.eye(H) - np.eye(H, k=-1)   # steer rate: first-diff on u

    self.w_lat = np.sqrt(lat_weight)
    self.w_jerk = np.sqrt(jerk_weight) / DEL_T
    self.w_rate = np.sqrt(steer_rate_weight)
    self.w_ref = np.sqrt(steer_ref_weight)

    # The QP's prediction gain must match reality, which rises with |lataccel|.
    # Precompute one solver at the normal gain and one at the high-regime gain;
    # update() blends them by upcoming target aggressiveness so the optimizer
    # isn't blind to the higher gain on aggressive maneuvers. The cost vector c is
    # gain-independent (it uses only Apow/Rc/E), so this is just a 2nd matmul.
    self.sched_qp = sched_qp
    b_hi = 1.96 * (1 - a)                 # b for the high-lataccel regime gain
    self.P_lo = self._build_solver(a, b)
    self.P_hi = self._build_solver(a, b_hi) if sched_qp else self.P_lo

  def _build_solver(self, a, b):
    """Precompute P = (M^T M)^-1 M^T for the steer->lat gain implied by (a, b)."""
    H = self.H
    G = np.zeros((H, H))
    for i in range(H):
      for j in range(i + 1):
        G[i, j] = (a ** (i - j)) * b
    M = np.vstack([
      self.w_lat * G,          # tracking residual
      self.w_jerk * (self.E @ G),  # jerk residual
      self.w_rate * self.D,    # steer-rate residual
      self.w_ref * np.eye(H),  # steer-reference residual (anchor to u_ss)
    ])
    return np.linalg.solve(M.T @ M, M.T)

  def _pad(self, seq, fill):
    seq = list(seq[:self.H])
    if not seq:
      seq = [fill]
    if len(seq) < self.H:
      seq = seq + [seq[-1]] * (self.H - len(seq))
    return np.array(seq)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    H = self.H

    # Optional EMA denoising of the measured lataccel used as the QP's start state
    if self.lat_filt is None:
      self.lat_filt = current_lataccel
    self.lat_filt = self.lat_filter * current_lataccel + (1 - self.lat_filter) * self.lat_filt
    lat0 = self.lat_filt

    # future_plan[0] is the target for the next step (which u[0] produces).
    target_seq = self._pad(future_plan.lataccel, target_lataccel)
    roll_seq = self._pad(future_plan.roll_lataccel, state.roll_lataccel)

    # Free response (zero steer):  f = Apow*lat0 + Rc @ roll
    f = self.Apow * lat0 + self.Rc @ roll_seq

    # Constant parts of each residual block
    e0 = np.zeros(H)
    e0[0] = -lat0              # lat0 -> lat[1] jerk transition
    d0 = np.zeros(H)
    d0[0] = self.prev_steer    # prev -> u[0] steer-rate transition

    # Steady-state feed-forward steer the QP is anchored to. Use the
    # magnitude-scheduled gain so aggressive targets aren't over-commanded into
    # saturation; fall back to the constant gain when scheduling is disabled.
    if self.sched_gain:
      gains = np.interp(np.abs(target_seq), self._gain_lat, self._gain_val)
    else:
      gains = self.steer_ss_gain
    u_ss = (target_seq - self.roll_ss_gain * roll_seq) / gains

    c1 = self.w_lat * (target_seq - f)
    c2 = -self.w_jerk * (self.E @ f + e0)
    c3 = self.w_rate * d0
    c4 = self.w_ref * u_ss
    c = np.concatenate([c1, c2, c3, c4])

    # Blend the normal- and high-gain solvers by how aggressive the upcoming
    # maneuver is, so the QP's prediction gain tracks the plant's |lat|-dependent
    # gain (matches the feed-forward schedule). agg: 0 below |lat|=1.5, 1 above 2.5.
    if self.sched_qp:
      agg = float(np.clip((np.max(np.abs(target_seq)) - 1.5) / 1.0, 0.0, 1.0))
      u = (1.0 - agg) * (self.P_lo @ c) + agg * (self.P_hi @ c)
    else:
      u = self.P_lo @ c

    # Integral correction on the current tracking error (kills steady-state lag)
    self.error_integral += (target_lataccel - current_lataccel)
    integral_term = self.i_gain * self.error_integral
    qp_steer = u[0] + integral_term

    # Soft start: blend feed-forward -> full QP over the first few control steps
    ctrl_idx = self.call_count - self.takeover_call
    self.call_count += 1
    if ctrl_idx < 0:
      alpha = 1.0  # warm-up: output is discarded by the sim anyway
    else:
      alpha = min(1.0, ctrl_idx / max(self.soft_steps, 1))
    ff_steer = u_ss[0]  # smooth steady-state holding steer (incl. roll comp)
    steer_unclipped = (1.0 - alpha) * ff_steer + alpha * qp_steer

    steer = float(np.clip(steer_unclipped, STEER_MIN, STEER_MAX))

    # Back-calculation anti-windup: when the output saturates (e.g. a physically
    # unachievable target), remove the railed excess from the integral so it
    # cannot wind up and then violently unwind. This is what was blowing up jerk
    # on the high-lataccel outlier segments. A magnitude clamp backstops it.
    self.error_integral -= (steer_unclipped - steer) / max(self.i_gain, 1e-6)
    self.error_integral = float(np.clip(self.error_integral,
                                        -self.i_clip / max(self.i_gain, 1e-6),
                                        self.i_clip / max(self.i_gain, 1e-6)))

    self.prev_steer = steer
    return steer
