"""Template tests. Every composed example inherits these and must pass them in CI.

They encode what every submittable repo should satisfy: the manifest parses and names a
loadable, instantiable agent class with the required interface, and that agent can actually
drive a few steps of the synced environment headlessly. They pass on the bare template because
it ships a small working starting agent, so a fresh clone is green out of the box; they keep
gating composed examples in CI and a student's own edits locally.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import cast

import agent
from sandbox.env import META, make_env
from sandbox.harness.environment import resolve_parameters
from sandbox.observation_types import ThreeBranchesObservation
from sandbox.play import load_agent, play_episode

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_manifest_parses_and_names_a_loadable_class():
    manifest = json.loads((REPO_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"entry_point", "class_name", "template_version"}
    assert isinstance(manifest["template_version"], int)
    agent = load_agent(REPO_ROOT)
    assert agent is not None


def test_three_step_headless_episode_runs():
    agent = load_agent(REPO_ROOT)
    env = make_env(resolve_parameters(META))
    try:
        score = play_episode(agent, env, seed=0, max_steps=3)
    finally:
        env.close()
    assert isinstance(score, float)


def test_selected_task_lasts_30_to_45_ticks():
    env = make_env(resolve_parameters(META, {"seat_plan": "cast_5", "daynight": False}))
    try:
        observations, _ = env.reset(seed=11)
        observation = cast(ThreeBranchesObservation, observations["player_1"])
        resident = agent.Agent()
        resident.reset(11, observation)
        resident.act(observation)
    finally:
        env.close()

    assert resident._task is not None
    assert 30 <= resident._task_remaining <= 45


def test_visitor_chat_uses_background_llm_and_replies_to_sender(monkeypatch):
    class FakeBackgroundLLM:
        def __init__(self):
            self.pending = False
            self.replies = deque()
            self.requests = []

        @property
        def requesting(self):
            return self.pending

        def request(self, **kwargs):
            self.requests.append(kwargs)
            self.pending = True
            return True

        def response(self):
            if not self.replies:
                return None
            self.pending = False
            return self.replies.popleft()

    monkeypatch.setattr(agent, "BackgroundLLM", FakeBackgroundLLM)
    env = make_env(resolve_parameters(META))
    try:
        observations, _ = env.reset(seed=12)
        observation = cast(ThreeBranchesObservation, observations["player_1"])
        resident = agent.Agent()
        resident.reset(12, observation)
    finally:
        env.close()

    assert resident.chat([{"from": "player_0", "text": "Hello!"}]) == []
    llm = cast(FakeBackgroundLLM, resident._llm)
    assert llm.requests[0]["messages"][-1] == {
        "role": "user",
        "content": "Hello!",
    }

    llm.replies.append("Good day!")
    assert resident.chat([]) == [{"to": "player_0", "text": "Good day!"}]
