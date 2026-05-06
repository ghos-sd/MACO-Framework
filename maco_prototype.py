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
# =========================
# MACO v2.2 - Improved Prototype
# =========================
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum
import itertools

# =========================
# Medical Knowledge Base (simplified)
# =========================
DRUG_DB = {
    "DrugA": {
        "indications": ["heart_failure"],
        "contraindications": ["renal_impairment", "allergy_penicillin"],
        "risks": {"renal_toxicity": 0.8, "hypotension": 0.3},
        "interactions": {"DrugC": "increased_renal_damage"},
        "evidence_level": "A",
        "mechanism": "ACE_inhibitor"
    },
    "DrugB": {
        "indications": ["heart_failure", "hypertension"],
        "contraindications": ["severe_hypotension"],
        "risks": {"hypotension": 0.6, "bradycardia": 0.2},
        "interactions": {"DrugD": "additive_hypotension"},
        "evidence_level": "A",
        "mechanism": "beta_blocker"
    },
    "DrugC": {
        "indications": ["infection_bacterial"],
        "contraindications": ["renal_impairment", "allergy_penicillin"],
        "risks": {"renal_toxicity": 0.9, "gi_upset": 0.4},
        "interactions": {"DrugA": "increased_renal_damage"},
        "evidence_level": "B",
        "mechanism": "aminoglycoside"
    },
    "DrugD": {
        "indications": ["infection_bacterial"],
        "contraindications": ["allergy_sulfa"],
        "risks": {"hypotension": 0.5, "allergic_reaction": 0.3},
        "interactions": {"DrugB": "additive_hypotension"},
        "evidence_level": "B",
        "mechanism": "sulfonamide"
    },
    "DrugE": {
        "indications": ["diabetes_type2"],
        "contraindications": ["renal_impairment", "liver_failure"],
        "risks": {"hypoglycemia": 0.7, "weight_gain": 0.4},
        "interactions": {},
        "evidence_level": "A",
        "mechanism": "sulfonylurea"
    },
    "DrugF": {
        "indications": ["heart_failure", "renal_protection"],
        "contraindications": [],
        "risks": {"hyperkalemia": 0.5, "cough": 0.2},
        "interactions": {},
        "evidence_level": "A",
        "mechanism": "ARB"
    }
}

# Organ risk mapping: which organs are affected by a risk
ORGAN_RISK_MAP = {
    "renal_toxicity": "kidney",
    "hypotension": "cardiovascular",
    "bradycardia": "cardiovascular",
    "hyperkalemia": "kidney",
    "hypoglycemia": "metabolic",
    "gi_upset": "gi",
    "allergic_reaction": "immune",
    "cough": "respiratory",
    "weight_gain": "metabolic"
}

# =========================
# Patient Data Model
# =========================
@dataclass
class Patient:
    id: str
    conditions: List[str]  # e.g., ["heart_failure", "kidney_disease"]
    allergies: List[str]   # e.g., ["penicillin", "sulfa"]
    medications: List[str] # current active medications
    lab_results: Dict[str, float]  # e.g., {"creatinine": 3.1, "egfr": 25}
    vitals: Dict[str, float]      # e.g., {"bp_systolic": 90, "heart_rate": 110}
    age: int = 65
    weight_kg: float = 70.0

    def has_condition(self, condition: str) -> bool:
        return condition in self.conditions

    def has_allergy(self, allergen: str) -> bool:
        return any(allergen in allergy.lower() for allergy in self.allergies)

    def has_medication(self, drug: str) -> bool:
        return drug in self.medications

# =========================
# Agent Proposal
# =========================
@dataclass
class Proposal:
    agent_name: str
    treatment: str
    confidence: float
    evidence_level: str
    risks: List[str]  # list ofrisk keys
    mechanism: str
    alternative_available: bool = False
    alternative_to: Optional[str] = None

# =========================
# Conflict & Scoring Helpers
# =========================
EVIDENCE_WEIGHTS = {"A": 1.0, "B": 0.7, "C": 0.4}

def organ_conflict(risk1: str, risk2: str) -> float:
    """Return a conflict score between two risks based on organ overlap."""
    organ1 = ORGAN_RISK_MAP.get(risk1)
    organ2 = ORGAN_RISK_MAP.get(risk2)
    if organ1 and organ2 and organ1 == organ2:
        return 0.8  # same organ system -> high conflict
    elif organ1 and organ2:
        # cross-organ interaction (e.g., kidney and cardiovascular are linked)
        related_pairs = {("kidney", "cardiovascular"): 0.6, ("cardiovascular", "respiratory"): 0.5}
        if (organ1, organ2) in related_pairs or (organ2, organ1) in related_pairs:
            return related_pairs.get((organ1, organ2), 0.3)
    return 0.2  # low cross-organ risk

# =========================
# HCA (Historical Context Agent)
# =========================
class HCA:
    @staticmethod
    def validate(patient: Patient, proposal: Proposal) -> Tuple[bool, str]:
        # Check allergies (simplified: match treatment name to allergy list)
        # More realistic: check drug class
        for allergy in patient.allergies:
            # This is a crude match; in reality we'd map drugs to allergen categories
            if allergy.lower() in proposal.treatment.lower():
                return False, f"ALLERGY_CONFLICT: {proposal.treatment} matches allergy {allergy}"
        
        # Check contraindications for patient conditions
        drug_info = DRUG_DB.get(proposal.treatment)
        if drug_info:
            contraindications = drug_info.get("contraindications", [])
            if "renal_impairment" in contraindications and patient.has_condition("kidney_disease"):
                if patient.lab_results.get("creatinine", 0) > 2.5:
                    return False, f"CONTRAINDICATION: {proposal.treatment} contraindicated in severe renal impairment (creatinine {patient.lab_results['creatinine']})"
            if "severe_hypotension" in contraindications and patient.vitals.get("bp_systolic", 120) < 100:
                return False, f"CONTRAINDICATION: {proposal.treatment} contraindicated in hypotension (systolic BP {patient.vitals['bp_systolic']})"
            
            # Check drug-drug interactions with current medications
            for current_med in patient.medications:
                interactions = drug_info.get("interactions", {})
                if current_med in interactions:
                    severity = interactions[current_med]
                    if "renal" in severity.lower() or "additive_hypotension" in severity.lower():
                        return False, f"DRUG_INTERACTION: {proposal.treatment} + {current_med} -> {severity}"
        
        return True, "SAFE"

# =========================
# Specialized Agents (simulating domain reasoning)
# =========================
class CardiologyAgent:
    def propose(self, patient: Patient) -> Optional[Proposal]:
        # If heart failure, suggest ACEi or ARB
        if patient.has_condition("heart_failure"):
            # Check renal function to decide
            if patient.lab_results.get("creatinine", 0) > 2.0:
                # Renal impairment: avoid DrugA (ACEi) -> try ARB
                return Proposal("Cardiology", "DrugF", 0.85, "A", ["hyperkalemia"], "ARB")
            else:
                # Standard ACEi
                return Proposal("Cardiology", "DrugA", 0.92, "A", ["renal_toxicity", "hypotension"], "ACE_inhibitor")
        return None

class NephrologyAgent:
    def propose(self, patient: Patient) -> Optional[Proposal]:
        if patient.has_condition("kidney_disease"):
            # Protect kidneys: avoid nephrotoxins
            # If infection present, need to suggest renal-safe antibiotic later
            return Proposal("Nephrology", "DrugF", 0.88, "A", ["hyperkalemia"], "ARB", alternative_available=True)
        return None

class InfectiousDiseaseAgent:
    def propose(self, patient: Patient) -> Optional[Proposal]:
        # Simulate: presence of infection condition
        if any("infection" in c for c in patient.conditions):
            # Check renal function to pick antibiotic
            if patient.lab_results.get("creatinine", 0) > 2.0:
                # Avoid aminoglycosides (DrugC) -> use sulfonamide (DrugD) if no sulfa allergy
                return Proposal("ID", "DrugD", 0.78, "B", ["hypotension", "allergic_reaction"], "sulfonamide")
            else:
                return Proposal("ID", "DrugC", 0.90, "B", ["renal_toxicity", "gi_upset"], "aminoglycoside")
        return None

class EndocrinologyAgent:
    def propose(self, patient: Patient) -> Optional[Proposal]:
        if patient.has_condition("diabetes_type2"):
            if patient.lab_results.get("creatinine", 0) > 2.0:
                # Renal impairment: avoid sulfonylureas -> propose insulin or metformin (not modeled)
                # Here, just skip or propose a safer alternative (simulated)
                return None
            else:
                return Proposal("Endocrinology", "DrugE", 0.82, "A", ["hypoglycemia", "weight_gain"], "sulfonylurea")
        return None

# =========================
# Conflict Matrix Engine
# =========================
class ConflictEngine:
    @staticmethod
    def compute_conflict(p1: Proposal, p2: Proposal) -> float:
        # Base conflict from therapeutic distance (difference in mechanisms? use confidence difference as proxy)
        distance = abs(p1.confidence - p2.confidence)
        
        # Cross-risk organ conflict
        contra_risk = 0.0
        for r1 in p1.risks:
            for r2 in p2.risks:
                contra_risk += organ_conflict(r1, r2)
        
        # Drug interaction penalty (if both drugs are proposed and interact)
        drug_info1 = DRUG_DB.get(p1.treatment, {})
        interactions = drug_info1.get("interactions", {})
        if p2.treatment in interactions:
            contra_risk += 0.9  # direct interaction
        
        return round(distance + contra_risk, 3)

# =========================
# Orchestrator with Negotiation
# =========================
class Orchestrator:
    def __init__(self, agents: List, hca: HCA, conflict_threshold: float = 0.8):
        self.agents = agents
        self.hca = hca
        self.conflict_engine = ConflictEngine()
        self.conflict_threshold = conflict_threshold
        self.alpha = 0.4
        self.beta = 0.35
        self.gamma = 0.25

    def score_proposal(self, proposal: Proposal) -> float:
        evidence_weight = EVIDENCE_WEIGHTS.get(proposal.evidence_level, 0.5)
        risk_penalty = sum(ORGAN_RISK_MAP.get(r, 0.5) for r in proposal.risks) * 0.3
        score = (self.alpha * proposal.confidence) + (self.beta * evidence_weight) - (self.gamma * risk_penalty)
        return round(score, 3)

    def check_emergency(self, patient: Patient) -> bool:
        # Simple criteria: severe hypotension, tachycardia, or critical lab values
        if patient.vitals.get("bp_systolic", 120) < 80 and patient.vitals.get("heart_rate", 70) > 120:
            return True
        return False

    def evaluate(self, patient: Patient):
        print("\n" + "="*50)
        print("MACO v2.2 CLINICAL ORCHESTRATION")
        print("="*50)
        print(f"Patient: {patient.id}")
        print(f"Conditions: {patient.conditions}")
        print(f"Allergies: {patient.allergies}")
        print(f"Medications: {patient.medications}")
        print(f"Labs: {patient.lab_results}")
        print(f"Vitals: {patient.vitals}\n")

        if self.check_emergency(patient):
            print("[EMERGENCY] Critical vitals detected – activating interrupt protocol.")
            print("  -> Immediate stabilization: IV fluids, vasopressors, empirical broad-spectrum antibiotics (if infection suspected).")
            print("  Bypassing extended negotiation.\n")
            return

        # Phase 1: collect initial proposals from agents
        proposals = []
        for agent in self.agents:
            try:
                prop = agent.propose(patient)
                if prop:
                    proposals.append(prop)
            except Exception as e:
                pass

        # Phase 2: HCA validation (deterministic safety)
        valid_proposals = []
        for prop in proposals:
            valid, reason = self.hca.validate(patient, prop)
            if not valid:
                print(f"[VETO] {prop.agent_name} : {prop.treatment}  | Reason: {reason}\n")
            else:
                valid_proposals.append((prop, reason))
                print(f"[OK]   {prop.agent_name} : {prop.treatment}  | {reason}")

        if not valid_proposals:
            print("\nNo valid proposals after safety check.")
            return

        # Phase 3: Build conflict matrix
        print("\n" + "-"*40)
        print("CONFLICT MATRIX")
        matrix = {}
        for i, (p1, _) in enumerate(valid_proposals):
            for j, (p2, _) in enumerate(valid_proposals):
                if i < j:
                    conflict = self.conflict_engine.compute_conflict(p1, p2)
                    matrix[(i, j)] = conflict
                    print(f"  {p1.agent_name}/{p1.treatment} vs {p2.agent_name}/{p2.treatment}: {conflict}")

        total_conflict = sum(matrix.values())
        print(f"Total Conflict Score: {total_conflict:.3f}")

        # Phase 4: Scoring initial proposals
        scored = [(p, self.score_proposal(p)) for p, _ in valid_proposals]
        print("\n" + "-"*40)
        print("INITIAL SCORES")
        for prop, score in sorted(scored, key=lambda x: x[1], reverse=True):
            print(f"  {prop.agent_name} | {prop.treatment} | Score={score}")

        # Phase 5: Negotiation if conflict exceeds threshold
        if total_conflict > self.conflict_threshold:
            print("\n[!] High conflict – initiating negotiation loop...")
            # Try to find alternatives: ask each agent to propose an alternative if available
            alternative_proposals = []
            for agent in self.agents:
                if hasattr(agent, "propose_alternative"):
                    alt = agent.propose_alternative(patient, [p for p,_ in valid_proposals])
                    if alt:
                        valid_alt, _ = self.hca.validate(patient, alt)
                        if valid_alt:
                            alternative_proposals.append(alt)
            if alternative_proposals:
                print("  Alternatives proposed:")
                for alt in alternative_proposals:
                    print(f"    {alt.agent_name}: {alt.treatment}")
                # Re-run conflict matrix with alternatives
                # Simplified: replace the highest-risk proposal with its alternative
                # In a real implementation, we'd do combinatorial optimization.
                # Here we just append and re-evaluate.
                all_combos = valid_proposals + [(p, "ALTERNATIVE") for p in alternative_proposals]
                best_combo = None
                min_conflict = float('inf')
                for combo in itertools.combinations(all_combos, r=len(valid_proposals)):
                    total_c = 0
                    for pair in itertools.combinations(combo, 2):
                        total_c += self.conflict_engine.compute_conflict(pair[0][0], pair[1][0])
                    if total_c < min_conflict:
                        min_conflict = total_c
                        best_combo = combo
                if best_combo:
                    print(f"  → Optimized conflict: {min_conflict:.3f}")
                    # Use best combo as final candidates
                    final_candidates = [p for p,_ in best_combo]
                else:
                    final_candidates = [p for p,_ in valid_proposals]
            else:
                print("  No alternatives found. Proceeding with escalation note.")
                final_candidates = [p for p,_ in valid_proposals]
        else:
            final_candidates = [p for p,_ in valid_proposals]

        # Final scoring
        final_scores = [(p, self.score_proposal(p)) for p in final_candidates]
        final_scores.sort(key=lambda x: x[1], reverse=True)
        print("\n" + "-"*40)
        print("FINAL RECOMMENDATION")
        winner = final_scores[0]
        print(f"  Treatment: {winner[0].treatment}")
        print(f"  Recommended by: {winner[0].agent_name} (evidence level {winner[0].evidence_level})")
        print(f"  Final Score: {winner[1]}")
        print(f"  Risks: {', '.join(winner[0].risks)}")
        # Rejected alternatives
        rejected = [f"{p.agent_name}/{p.treatment}" for p,_ in final_candidates if p!=winner[0]]
        if rejected:
            print(f"  Rejected alternatives: {', '.join(rejected)}")
        print("="*50 + "\n")

# =========================
# Simulation
# =========================
if __name__ == "__main__":
    # Create patient with complex profile
    patient = Patient(
        id="PX-001",
        conditions=["heart_failure", "kidney_disease", "infection_bacterial", "diabetes_type2"],
        allergies=["penicillin"],  # sensitive to penicillin
        medications=["metformin"],  # assume ongoing
        lab_results={"creatinine": 3.1, "egfr": 22, "potassium": 5.2},
        vitals={"bp_systolic": 105, "heart_rate": 88, "temperature": 38.2},
        age=72,
        weight_kg=68.5
    )

    # Instantiate agents
    agents = [
        CardiologyAgent(),
        NephrologyAgent(),
        InfectiousDiseaseAgent(),
        EndocrinologyAgent()
    ]
    hca = HCA()
    orchestrator = Orchestrator(agents, hca, conflict_threshold=1.2)
    orchestrator.evaluate(patient)
