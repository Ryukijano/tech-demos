"""Tiny GRU policy. Answer digits are predicted one at a time after the prompt."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from task import N_DIGITS, VOCAB


class DigitPolicy(nn.Module):
    def __init__(self, emb: int, hidden: int, vocab: int = VOCAB, n_digits: int = N_DIGITS) -> None:
        super().__init__()
        self.vocab = vocab
        self.n_digits = n_digits
        self.hidden = hidden
        self.emb = nn.Embedding(vocab, emb)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _prompt_logits(self, prompts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, hidden = self.gru(self.emb(prompts))
        return self.head(out[:, -1, :]), hidden

    def _digit_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(logits[:, : self.n_digits], dim=-1)

    def token_logprobs(self, prompts: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
        """Per-token log π(digit | prompt, previous digits). Shape [N, A]."""
        logits, hidden = self._prompt_logits(prompts)
        columns: list[torch.Tensor] = []
        for t in range(answers.size(1)):
            log_probs = self._digit_log_probs(logits)
            columns.append(log_probs.gather(1, answers[:, t : t + 1]).squeeze(1))
            step, hidden = self.gru(self.emb(answers[:, t]).unsqueeze(1), hidden)
            logits = self.head(step[:, 0, :])
        return torch.stack(columns, dim=1)

    @torch.no_grad()
    def sample_len(
        self,
        prompts: torch.Tensor,
        n_group: int,
        answer_len: int,
        generator: torch.Generator,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        flat = prompts.repeat_interleave(n_group, dim=0)
        logits, hidden = self._prompt_logits(flat)
        columns: list[torch.Tensor] = []
        for _ in range(answer_len):
            probs = torch.softmax(logits[:, : self.n_digits] / temperature, dim=-1)
            tok = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
            columns.append(tok)
            step, hidden = self.gru(self.emb(tok).unsqueeze(1), hidden)
            logits = self.head(step[:, 0, :])
        batch = prompts.size(0)
        return torch.stack(columns, dim=1).view(batch, n_group, answer_len)

    @torch.no_grad()
    def greedy(self, prompts: torch.Tensor, answer_len: int) -> torch.Tensor:
        logits, hidden = self._prompt_logits(prompts)
        columns: list[torch.Tensor] = []
        for _ in range(answer_len):
            tok = torch.argmax(logits[:, : self.n_digits], dim=-1)
            columns.append(tok)
            step, hidden = self.gru(self.emb(tok).unsqueeze(1), hidden)
            logits = self.head(step[:, 0, :])
        return torch.stack(columns, dim=1)
