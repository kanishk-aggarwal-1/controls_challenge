from . import BaseController
import numpy as np


# Steady-state steer needed to hold a given lataccel, from probe_dynamics.py:
#   lataccel_ss = B/(1-A) * steer + C/(1-A) * roll
# Inverting:  steer = (lataccel - ROLL_GAIN*roll) / STEER_GAIN
STEER_GAIN = 1.95   # B/(1-A): lataccel produced per unit steer at steady state
ROLL_GAIN = 1.34    # C/(1-A): lataccel produced per unit roll at steady state


class Controller(BaseController):
  """
  Feed-forward + PID controller.

  The feed-forward term inverts the (measured) plant model to directly command
  the steer that should hold the target lataccel, compensating for road roll.
  The PID then only has to correct the residual error, so its gains can be small
  and smooth -> low jerk.
  """
  def __init__(self,
               p=0.11, i=0.10, d=-0.01,
               ff_gain=1.0, lookahead=2):
    self.p = p
    self.i = i
    self.d = d
    self.ff_gain = ff_gain      # weight on the model-inverting feed-forward
    self.lookahead = lookahead  # steps ahead to aim the feed-forward (latency comp)

    self.error_integral = 0.0
    self.prev_error = 0.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    # --- Feed-forward: invert the plant to get the steer that holds the target ---
    # Aim slightly ahead to compensate for actuation lag.
    if len(future_plan.lataccel) > self.lookahead:
      ff_target = future_plan.lataccel[self.lookahead]
      ff_roll = future_plan.roll_lataccel[self.lookahead]
    else:
      ff_target = target_lataccel
      ff_roll = state.roll_lataccel

    feedforward = self.ff_gain * (ff_target - ROLL_GAIN * ff_roll) / STEER_GAIN

    # --- PID: correct the residual tracking error ---
    error = target_lataccel - current_lataccel
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error
    pid = self.p * error + self.i * self.error_integral + self.d * error_diff

    return feedforward + pid
