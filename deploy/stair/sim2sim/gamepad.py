"""PIE-compatible local gamepad control, with Stair's command limits."""

import math
import os
from dataclasses import dataclass
import numpy as np

GAMEPAD_AXIS = {
    "xbox": {"LX": 0, "LY": 1, "RX": 3, "RY": 4},
    "switch": {"LX": 0, "LY": 1, "RX": 2, "RY": 3},
}

GAMEPAD_BUTTON = {
    "xbox": {"A": 0, "START": 7},
    "switch": {"A": 0, "START": 11},
}


@dataclass(frozen=True)
class GamepadState:
    lx: float
    ly: float
    rx: float
    ry: float
    stop_pressed: bool
    reset_pressed: bool


def apply_deadzone(value: float, deadzone: float) -> float:
    """Apply a continuous deadzone while preserving full-scale endpoints."""
    if not math.isfinite(value):
        raise ValueError("Gamepad axis must be finite.")
    if not math.isfinite(deadzone) or not 0.0 <= deadzone < 1.0:
        raise ValueError("Gamepad deadzone must lie in [0, 1).")
    clipped = float(np.clip(value, -1.0, 1.0))
    if abs(clipped) <= deadzone:
        return 0.0
    return float(np.sign(clipped) * (abs(clipped) - deadzone) / (1.0 - deadzone))


def forward_speed_from_unit_input(
    value: float,
    *,
    max_speed: float,
    deadzone: float,
) -> float:
    if not math.isfinite(max_speed) or max_speed < 0.0:
        raise ValueError("Gamepad maximum speed must be finite and nonnegative.")
    return float(max(0.0, apply_deadzone(value, deadzone)) * max_speed)


def yaw_rate_from_unit_input(
    value: float,
    *,
    max_yaw_rate: float,
    deadzone: float,
) -> float:
    if not math.isfinite(max_yaw_rate) or max_yaw_rate < 0.0:
        raise ValueError("Gamepad maximum yaw rate must be finite and nonnegative.")
    return float(-apply_deadzone(value, deadzone) * max_yaw_rate)

class Gamepad:
    def __init__(self, device_id: int, gamepad_type: str):
        if gamepad_type not in GAMEPAD_AXIS:
            raise ValueError(f"Unsupported gamepad type {gamepad_type!r}.")
        os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")
        # SDL is only an input backend here. GLFW owns the MuJoCo window.
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("Gamepad control requires pygame: pip install pygame") from exc
        self._pygame = pygame
        self._axis, self._button = GAMEPAD_AXIS[gamepad_type], GAMEPAD_BUTTON[gamepad_type]
        pygame.display.init()
        pygame.joystick.init()
        self._joystick = None
        try:
            count = pygame.joystick.get_count()
            if count == 0:
                raise RuntimeError("No gamepad detected. Connect it before using --joystick.")
            if not 0 <= device_id < count:
                raise ValueError(f"Joystick {device_id} unavailable; detected {count} device(s).")
            self._joystick = pygame.joystick.Joystick(device_id)
            self._joystick.init()
            self._instance_id = self._joystick.get_instance_id()
            self._disconnected = False
            print(f"[GAMEPAD] {self._joystick.get_name()} ({gamepad_type}, device {device_id}, "
                  f"axes={self._joystick.get_numaxes()}, buttons={self._joystick.get_numbuttons()})")
        except Exception:
            self.close()
            raise

    def _get_axis(self, name: str) -> float:
        axis_id = self._axis[name]
        if axis_id >= self._joystick.get_numaxes():
            return 0.0
        return float(np.clip(self._joystick.get_axis(axis_id), -1.0, 1.0))

    def _get_button(self, name: str) -> bool:
        button_id = self._button[name]
        if button_id >= self._joystick.get_numbuttons():
            return False
        return bool(self._joystick.get_button(button_id))

    def sample(self):
        # pygame.joystick.Joystick has no get_attached(). Track SDL removal
        # events by instance ID (device indices can change after unplugging).
        removed = self._pygame.event.get([self._pygame.JOYDEVICEREMOVED])
        self._disconnected |= any(event.instance_id == self._instance_id for event in removed)
        if self._disconnected or not self._joystick.get_init():
            raise RuntimeError("Gamepad disconnected; stopping simulation.")
        return GamepadState(
            lx=self._get_axis("LX"), ly=-self._get_axis("LY"),
            rx=self._get_axis("RX"), ry=-self._get_axis("RY"),
            stop_pressed=self._get_button("A"), reset_pressed=self._get_button("START"),
        )

    def close(self):
        if self._joystick is not None:
            self._joystick.quit()
        self._pygame.joystick.quit()
        self._pygame.display.quit()


class GamepadVelocityController:
    def __init__(self, gamepad, *, max_speed=.8, max_yaw_rate=.8, deadzone=.1):
        forward_speed_from_unit_input(0, max_speed=max_speed, deadzone=deadzone)
        yaw_rate_from_unit_input(0, max_yaw_rate=max_yaw_rate, deadzone=deadzone)
        self.gamepad = gamepad
        self.max_speed, self.max_yaw_rate, self.deadzone = max_speed, max_yaw_rate, deadzone
        self._reset_held = False

    def update(self, command):
        try:
            state = self.gamepad.sample()
        except Exception:
            command.vx = command.vy = command.wz = 0.0
            raise
        command.vy = 0.0
        command.stop_requested = state.stop_pressed
        command.vx = 0.0 if state.stop_pressed else forward_speed_from_unit_input(
            state.ly, max_speed=self.max_speed, deadzone=self.deadzone,
        )
        command.wz = 0.0 if state.stop_pressed else yaw_rate_from_unit_input(
            state.rx, max_yaw_rate=self.max_yaw_rate, deadzone=self.deadzone,
        )
        if state.reset_pressed and not self._reset_held:
            command.reset_requested = True
        self._reset_held = state.reset_pressed

    def close(self):
        self.gamepad.close()
