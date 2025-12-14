# interaction.py
from dataclasses import dataclass
from typing import List, Literal, Dict
import math
import random
from agents import DyadAgent
from task import program_length
from lexicon import LexiconEntry

@dataclass
class GameLogEntry:
    round_idx: int
    tower_id: str
    target_program: str
    utterance: str
    guessed_program: str
    success: bool
    speaker_mode: str
    listener_mode: str

def run_one_episode(
    tower_id: str,
    architect: DyadAgent,
    builder: DyadAgent,
    towers_cfg: Dict[str, Dict],
    speaker_mode: Literal["literal", "pragmatic"] = "pragmatic",
    listener_mode: Literal["literal", "pragmatic"] = "pragmatic",
    length_prior_lambda: float = 0.8,
) -> GameLogEntry:
    """
    一轮 Builder–Architect 交互：
    1. 根据 tower_id 按长度先验随机选一个 target_program 作为 meaning m*。
    2. Architect 用 RSA 选 utterance u。
    3. Builder 用 L0/L1 解释，选 MAP program \hat m。
    4. 如果 \hat m 在 correct_programs 里，视为 success。
    5. 双方 observe(m*, u) 更新信念。
    """
    # 1) 选择目标 program（按长度先验采样）
    info = towers_cfg[tower_id]
    programs = info["correct_programs"]
    lengths = [program_length(p) for p in programs]
    weights = [math.exp(-length_prior_lambda * (L - 1)) for L in lengths]
    if sum(weights) <= 0:
        target_program = random.choice(programs)
    else:
        target_program = random.choices(programs, weights=weights, k=1)[0]
    correct_programs = set(info["correct_programs"])

    # 2) Architect 说话
    if speaker_mode == "literal":
        utterance = architect.produce_utterance_literal(target_program)
    else:
        utterance = architect.produce_utterance_pragmatic(target_program)

    # 3) Builder 听话
    post = builder.interpret(utterance, mode=listener_mode)
    guessed_program = max(post.items(), key=lambda kv: kv[1])[0]

    # 4) success 判定
    success = guessed_program in correct_programs

    # 5) 更新（这里用真实 target_program 更新）
    architect.observe(target_program, utterance)
    builder.observe(target_program, utterance)

    return GameLogEntry(
        round_idx=-1,
        tower_id=tower_id,
        target_program=target_program,
        utterance=utterance,
        guessed_program=guessed_program,
        success=success,
        speaker_mode=speaker_mode,
        listener_mode=listener_mode,
    )

def run_dyad(
    architect: DyadAgent,
    builder: DyadAgent,
    towers_cfg: Dict[str, Dict],
    tower_ids: List[str],
    n_rounds: int,
    speaker_mode: str = "pragmatic",
    listener_mode: str = "pragmatic",
    seed: int = 0,
    length_prior_lambda: float = 0.8,
) -> List[GameLogEntry]:
    """
    简单版：Architect 永远是 speaker，Builder 永远是 listener。
    每轮随机选择一个 tower。
    """
    random.seed(seed)
    logs: List[GameLogEntry] = []

    for t in range(n_rounds):
        tower_id = random.choice(tower_ids)
        entry = run_one_episode(
            tower_id=tower_id,
            architect=architect,
            builder=builder,
            towers_cfg=towers_cfg,
            speaker_mode=speaker_mode,
            listener_mode=listener_mode,
            length_prior_lambda=length_prior_lambda,
        )
        entry.round_idx = t
        logs.append(entry)

    return logs
