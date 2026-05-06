# =========================
# MACO v2.2 Prototype
# Multi-Agent Clinical Orchestration
# =========================

from dataclasses import dataclass
from typing import List, Dict
import math

# =========================
# Patient Context
# =========================

patient = {
    "name": "Patient-X",
    "conditions": ["heart_failure", "kidney_disease"],
    "allergies": ["DrugA"],
    "lab": {
        "creatinine": 3.1
    }
}

# =========================
# Treatment Proposal
# =========================

@dataclass
class Proposal:
    agent: str
    treatment: str
    confidence: float
    evidence_level: str
    risks: List[str]


# =========================
# Specialized Agents
# =========================

class CardiologyAgent:

    def propose(self, patient):

        return Proposal(
            agent="Cardiology",
            treatment="DrugA",
            confidence=0.91,
            evidence_level="A",
            risks=["renal_toxicity"]
        )


class NephrologyAgent:

    def propose(self, patient):

        return Proposal(
            agent="Nephrology",
            treatment="DrugB",
            confidence=0.84,
            evidence_level="B",
            risks=["low_bp"]
        )


# =========================
# HCA (Historical Context Agent)
# Deterministic Safety Layer
# =========================

class HCA:

    def validate(self, patient, proposal):

        if proposal.treatment in patient["allergies"]:
            return False, "ALLERGY_CONFLICT"

        return True, "SAFE"


# =========================
# Evidence Scoring
# =========================

EVIDENCE_MAP = {
    "A": 1.0,
    "B": 0.7,
    "C": 0.4
}


# =========================
# Conflict Matrix Engine
# =========================

class ConflictEngine:

    def compute_conflict(self, p1, p2):

        distance = abs(p1.confidence - p2.confidence)

        contra_risk = 0

        if "renal_toxicity" in p1.risks:
            contra_risk += 0.8

        if "low_bp" in p2.risks:
            contra_risk += 0.2

        total = distance + contra_risk

        return round(total, 3)


# =========================
# Orchestrator
# =========================

class Orchestrator:

    def __init__(self):

        self.hca = HCA()
        self.conflict_engine = ConflictEngine()

        self.alpha = 0.5
        self.beta = 0.4
        self.gamma = 0.3

    def score(self, proposal):

        evidence = EVIDENCE_MAP[proposal.evidence_level]

        risk_penalty = len(proposal.risks) * 0.5

        final_score = (
            (self.alpha * proposal.confidence)
            + (self.beta * evidence)
            - (self.gamma * risk_penalty)
        )

        return round(final_score, 3)

    def evaluate(self, patient, proposals):

        print("\n========== MACO ORCHESTRATION ==========\n")

        valid_proposals = []

        # Safety Validation
        for proposal in proposals:

            valid, reason = self.hca.validate(patient, proposal)

            if not valid:

                print(f"[VETO] {proposal.agent} -> {proposal.treatment}")
                print(f"Reason: {reason}\n")

                continue

            valid_proposals.append(proposal)

        # Conflict Analysis
        print("========== CONFLICT MATRIX ==========\n")

        for i in range(len(valid_proposals)):
            for j in range(i + 1, len(valid_proposals)):

                c = self.conflict_engine.compute_conflict(
                    valid_proposals[i],
                    valid_proposals[j]
                )

                print(
                    f"{valid_proposals[i].agent} vs "
                    f"{valid_proposals[j].agent}"
                )

                print(f"Conflict Score = {c}\n")

        # Scoring
        print("========== FINAL SCORING ==========\n")

        ranked = []

        for proposal in valid_proposals:

            s = self.score(proposal)

            ranked.append((proposal, s))

            print(
                f"{proposal.agent} | "
                f"{proposal.treatment} | "
                f"Score = {s}"
            )

        ranked.sort(key=lambda x: x[1], reverse=True)

        print("\n========== FINAL DECISION ==========\n")

        winner = ranked[0]

        print(
            f"Selected Treatment: {winner[0].treatment}"
        )

        print(
            f"Recommended By: {winner[0].agent}"
        )

        print(
            f"Final Score: {winner[1]}"
        )


# =========================
# Simulation
# =========================

cardio = CardiologyAgent()
nephro = NephrologyAgent()

proposals = [
    cardio.propose(patient),
    nephro.propose(patient)
]

maco = Orchestrator()

maco.evaluate(patient, proposals)
