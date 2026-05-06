"""
===========================================================================
MACO v2.2 – Multi-Agent Clinical Orchestration (Refactored Prototype)
===========================================================================
"Clinical safety emerges from structured conflict between constrained expert domains."

This prototype implements a deterministic, multi-agent clinical reasoning framework
with a focus on safety, explainability, and auditability. All components are
modular, research-oriented, and fully typed.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from enum import Enum, auto
import itertools

# ===========================================================================
# 1. ENUMS & CONSTANTS
# ===========================================================================

class Severity(Enum):
    """Standard severity scale for clinical risks."""
    CRITICAL = 5   # Immediate life‑threatening danger
    HIGH = 4       # Major organ risk or severe interaction
    MODERATE = 3   # Important, requiring mitigation
    LOW = 2        # Minor concern
    NEGLIGIBLE = 1 # Negligible risk

class EvidenceLevel(Enum):
    A = 1.0         # Meta‑analyses, RCTs
    B = 0.7         # Well‑designed non‑randomised studies
    C = 0.4         # Expert consensus / case reports
    D = 0.2         # Preclinical or anecdotal

class ContraindicationType(Enum):
    ALLERGY = auto()
    ORGAN_FAILURE = auto()
    DRUG_INTERACTION = auto()
    HEMODYNAMIC_INSTABILITY = auto()
    LAB_THRESHOLD = auto()

# Organ systems for cross‑organ conflict
class OrganSystem(Enum):
    CARDIOVASCULAR = "cardiovascular"
    RENAL = "renal"
    HEPATIC = "hepatic"
    RESPIRATORY = "respiratory"
    NEUROLOGICAL = "neurological"
    METABOLIC = "metabolic"
    IMMUNE = "immune"
    GI = "gastrointestinal"

# Mapping from risk string to (severity_value, primary_organ_system)
RISK_SEVERITY_MAP: Dict[str, Tuple[float, OrganSystem]] = {
    "renal_toxicity":      (0.9, OrganSystem.RENAL),
    "hepatotoxicity":      (0.9, OrganSystem.HEPATIC),
    "cardiotoxicity":      (0.95, OrganSystem.CARDIOVASCULAR),
    "neurotoxicity":       (0.85, OrganSystem.NEUROLOGICAL),
    "hypotension":         (0.7, OrganSystem.CARDIOVASCULAR),
    "bradycardia":         (0.6, OrganSystem.CARDIOVASCULAR),
    "hyperkalemia":        (0.8, OrganSystem.RENAL),
    "hypoglycemia":        (0.75, OrganSystem.METABOLIC),
    "gi_upset":            (0.4, OrganSystem.GI),
    "allergic_reaction":   (0.9, OrganSystem.IMMUNE),
    "weight_gain":         (0.3, OrganSystem.METABOLIC),
    "cough":               (0.2, OrganSystem.RESPIRATORY),
}

# Organ cross‑interaction penalty (additive when two different organs interact)
ORGAN_CROSS_PENALTY: Dict[Tuple[OrganSystem, OrganSystem], float] = {
    (OrganSystem.RENAL, OrganSystem.CARDIOVASCULAR): 0.6,
    (OrganSystem.CARDIOVASCULAR, OrganSystem.RENAL): 0.6,
    (OrganSystem.HEPATIC, OrganSystem.RENAL): 0.5,
    (OrganSystem.RENAL, OrganSystem.HEPATIC): 0.5,
    (OrganSystem.CARDIOVASCULAR, OrganSystem.RESPIRATORY): 0.5,
    (OrganSystem.RESPIRATORY, OrganSystem.CARDIOVASCULAR): 0.5,
    # Default cross‑organ influence handled in code (low)
}

# ===========================================================================
# 2. DRUG DATABASE (simplified pharmacopoeia)
# ===========================================================================

@dataclass
class DrugInfo:
    name: str
    indications: List[str]
    contraindications: Dict[str, ContraindicationType]  # condition -> type
    risks: Dict[str, float]          # risk -> internal severity (0‑1)
    interactions: Dict[str, str]     # other_drug -> description (severity inferred)
    evidence_level: EvidenceLevel
    mechanism: str
    guideline_source: str = "NICE"
    guideline_version: str = "2024"
    # Allergy cross‑reactivity mapping (drug class -> allergen)
    allergy_class: Optional[str] = None  # e.g., "penicillin", "sulfonamide"

# Example drugs (partial)
DRUG_DB: Dict[str, DrugInfo] = {
    "Lisinopril": DrugInfo(
        name="Lisinopril",
        indications=["heart_failure", "hypertension"],
        contraindications={
            "renal_impairment_severe": ContraindicationType.ORGAN_FAILURE,
            "angioedema": ContraindicationType.ALLERGY,
        },
        risks={"renal_toxicity": 0.8, "hypotension": 0.7, "hyperkalemia": 0.7},
        interactions={"Spironolactone": "severe_hyperkalemia", "Losartan": "additive_renal_risk"},
        evidence_level=EvidenceLevel.A,
        mechanism="ACE_inhibitor",
        allergy_class="ace_inhibitor",
    ),
    "Carvedilol": DrugInfo(
        name="Carvedilol",
        indications=["heart_failure", "hypertension"],
        contraindications={
            "severe_hypotension": ContraindicationType.HEMODYNAMIC_INSTABILITY,
            "bradycardia_severe": ContraindicationType.HEMODYNAMIC_INSTABILITY,
        },
        risks={"hypotension": 0.8, "bradycardia": 0.6, "fatigue": 0.3},
        interactions={"Verapamil": "additive_bradycardia"},
        evidence_level=EvidenceLevel.A,
        mechanism="beta_blocker",
    ),
    "Losartan": DrugInfo(
        name="Losartan",
        indications=["heart_failure", "renal_protection", "hypertension"],
        contraindications={
            "renal_artery_stenosis": ContraindicationType.ORGAN_FAILURE,
        },
        risks={"hyperkalemia": 0.6, "hypotension": 0.5},
        interactions={"Lisinopril": "additive_renal_risk"},
        evidence_level=EvidenceLevel.A,
        mechanism="ARB",
    ),
    "Gentamicin": DrugInfo(
        name="Gentamicin",
        indications=["infection_bacterial"],
        contraindications={
            "renal_impairment_severe": ContraindicationType.ORGAN_FAILURE,
            "allergy_aminoglycoside": ContraindicationType.ALLERGY,
        },
        risks={"renal_toxicity": 0.95, "ototoxicity": 0.7},
        interactions={"Furosemide": "increased_ototoxicity"},
        evidence_level=EvidenceLevel.B,
        mechanism="aminoglycoside",
        allergy_class="aminoglycoside",
    ),
    "Ceftriaxone": DrugInfo(
        name="Ceftriaxone",
        indications=["infection_bacterial"],
        contraindications={
            "allergy_cephalosporin": ContraindicationType.ALLERGY,
        },
        risks={"gi_upset": 0.3, "allergic_reaction": 0.4},
        interactions={},
        evidence_level=EvidenceLevel.A,
        mechanism="cephalosporin",
        allergy_class="cephalosporin",
    ),
    "Metformin": DrugInfo(
        name="Metformin",
        indications=["diabetes_type2"],
        contraindications={
            "renal_impairment_severe": ContraindicationType.ORGAN_FAILURE,
            "lactic_acidosis_risk": ContraindicationType.LAB_THRESHOLD,
        },
        risks={"gi_upset": 0.5, "lactic_acidosis": 0.9},
        interactions={"Contrast_dye": "lactic_acidosis_risk"},
        evidence_level=EvidenceLevel.A,
        mechanism="biguanide",
    ),
    "Furosemide": DrugInfo(
        name="Furosemide",
        indications=["fluid_overload", "heart_failure"],
        contraindications={
            "severe_hypotension": ContraindicationType.HEMODYNAMIC_INSTABILITY,
        },
        risks={"hypotension": 0.6, "electrolyte_imbalance": 0.7},
        interactions={"Gentamicin": "increased_ototoxicity"},
        evidence_level=EvidenceLevel.A,
        mechanism="loop_diuretic",
    ),
}

# Drug‑allergy mapping: allergen name → set of drug names (or classes)
ALLERGY_DRUG_MAP: Dict[str, Set[str]] = {
    "penicillin": {"Penicillin_V", "Amoxicillin", "Ampicillin"},  # not in our DB but illustrative
    "sulfonamide": {"Sulfamethoxazole"},
    "ace_inhibitor": {"Lisinopril", "Enalapril"},
    "aminoglycoside": {"Gentamicin", "Tobramycin"},
    "cephalosporin": {"Ceftriaxone", "Cefazolin"},
}

# ===========================================================================
# 3. PATIENT & CLINICAL DATA MODELS
# ===========================================================================

@dataclass
class VitalSigns:
    systolic_bp: float      # mmHg
    diastolic_bp: float     # mmHg
    heart_rate: float       # bpm
    respiratory_rate: float # breaths/min
    temperature_c: float    # Celsius
    spo2: float             # % (optional, could be None)

@dataclass
class LabResults:
    creatinine: float       # mg/dL
    egfr: float             # mL/min/1.73m² (calculated)
    potassium: float        # mmol/L
    sodium: float           # mmol/L
    wbc: Optional[float] = None
    lactate: Optional[float] = None

@dataclass
class Patient:
    id: str
    age: int
    weight_kg: float
    conditions: List[str]           # e.g., ["heart_failure", "kidney_disease"]
    allergies: List[str]            # e.g., ["penicillin", "sulfa"]
    medications: List[str]          # current active medications
    vitals: VitalSigns
    labs: LabResults

    def has_condition(self, condition: str) -> bool:
        return condition in self.conditions

    def has_any_allergy_class(self, classes: List[str]) -> bool:
        """Check if the patient has an allergy that cross‑reacts with any of the given drug classes."""
        for allergy in self.allergies:
            for cls in classes:
                if allergy.lower() in cls.lower() or cls.lower() in allergy.lower():
                    return True
        return False

    @property
    def renal_stage(self) -> str:
        """Interpret eGFR into CKD stage."""
        egfr = self.labs.egfr
        if egfr >= 90:
            return "Stage 1"
        elif egfr >= 60:
            return "Stage 2"
        elif egfr >= 45:
            return "Stage 3a"
        elif egfr >= 30:
            return "Stage 3b"
        elif egfr >= 15:
            return "Stage 4"
        else:
            return "Stage 5"

# ===========================================================================
# 4. PROPOSAL & GUIDELINE TRACEABILITY
# ===========================================================================

@dataclass
class Proposal:
    agent_name: str
    treatment: str           # drug name (from DRUG_DB)
    confidence: float        # agent’s own confidence (0‑1)
    evidence_level: EvidenceLevel
    risks: List[str]         # risk labels present in RISK_SEVERITY_MAP
    mechanism: str
    guideline_source: str = "NICE"
    guideline_version: str = "2024"
    alternative_to: Optional[str] = None  # if this is an alternative proposal

    def drug_info(self) -> Optional[DrugInfo]:
        return DRUG_DB.get(self.treatment)

# ===========================================================================
# 5. SAFETY LAYER (HCA & VALIDATION)
# ===========================================================================

@dataclass
class SafetyVeto:
    """Information about a safety veto."""
    proposal: Proposal
    reason: str
    severity: Severity
    veto_type: ContraindicationType

class HCA:
    """Deterministic Historical Context Agent – enforces immutable patient constraints."""

    @staticmethod
    def _check_allergy(patient: Patient, proposal: Proposal) -> Optional[SafetyVeto]:
        drug = proposal.drug_info()
        if not drug or not drug.allergy_class:
            return None
        # Map patient allergies to drug class
        for allergy in patient.allergies:
            # Check if patient allergy matches the drug's allergy class
            if allergy.lower() in drug.allergy_class.lower() or drug.allergy_class.lower() in allergy.lower():
                return SafetyVeto(
                    proposal,
                    f"Allergy conflict: drug class '{drug.allergy_class}' vs patient allergy '{allergy}'",
                    Severity.CRITICAL,
                    ContraindicationType.ALLERGY
                )
        return None

    @staticmethod
    def _check_organ_contraindications(patient: Patient, proposal: Proposal) -> Optional[SafetyVeto]:
        drug = proposal.drug_info()
        if not drug:
            return None
        contradictions = drug.contraindications
        # Renal impairment severity check
        if "renal_impairment_severe" in contradictions:
            if patient.labs.egfr < 30:
                return SafetyVeto(
                    proposal,
                    f"Renal contraindication: eGFR {patient.labs.egfr} (<30) with {drug.name}",
                    Severity.CRITICAL,
                    ContraindicationType.ORGAN_FAILURE
                )
        # Hemodynamic instability
        if "severe_hypotension" in contradictions or "bradycardia_severe" in contradictions:
            if patient.vitals.systolic_bp < 90:
                return SafetyVeto(
                    proposal,
                    f"Hemodynamic contraindication: SBP {patient.vitals.systolic_bp} mmHg too low for {drug.name}",
                    Severity.CRITICAL,
                    ContraindicationType.HEMODYNAMIC_INSTABILITY
                )
        return None

    @staticmethod
    def _check_drug_interactions(patient: Patient, proposal: Proposal) -> Optional[SafetyVeto]:
        drug = proposal.drug_info()
        if not drug:
            return None
        for med in patient.medications:
            if med in drug.interactions:
                interaction_desc = drug.interactions[med]
                # critical interactions lead to veto, others just increase risk
                if "severe" in interaction_desc or "life" in interaction_desc:
                    return SafetyVeto(
                        proposal,
                        f"Dangerous interaction: {drug.name} + {med} → {interaction_desc}",
                        Severity.CRITICAL,
                        ContraindicationType.DRUG_INTERACTION
                    )
        return None

    def validate(self, patient: Patient, proposal: Proposal) -> Tuple[bool, Optional[SafetyVeto]]:
        """Returns (is_safe, veto_info). If safe, veto_info is None."""
        for check in [self._check_allergy, self._check_organ_contraindications, self._check_drug_interactions]:
            veto = check(patient, proposal)
            if veto:
                return False, veto
        return True, None

# ===========================================================================
# 6. CLINICAL DOMAIN AGENTS
# ===========================================================================

class BaseClinicalAgent:
    """Abstract agent – must implement propose() and optionally propose_alternative()."""
    name: str = "BaseAgent"

    def propose(self, patient: Patient) -> Optional[Proposal]:
        raise NotImplementedError

    def propose_alternative(self, patient: Patient, current_conflicts: List[Proposal]) -> Optional[Proposal]:
        """Optionally generate a lower‑risk alternative."""
        return None

class CardiologyAgent(BaseClinicalAgent):
    name = "Cardiology"
    def propose(self, patient: Patient) -> Optional[Proposal]:
        if not patient.has_condition("heart_failure"):
            return None
        # Choose ACEi vs ARB vs beta‑blocker based on renal function
        if patient.labs.egfr < 30:
            drug = "Losartan"  # ARB preferred
        else:
            drug = "Lisinopril"  # ACEi first line
        info = DRUG_DB[drug]
        return Proposal(
            agent_name=self.name,
            treatment=drug,
            confidence=0.92,
            evidence_level=info.evidence_level,
            risks=list(info.risks.keys()),
            mechanism=info.mechanism,
            guideline_source=info.guideline_source,
            guideline_version=info.guideline_version
        )

    def propose_alternative(self, patient: Patient, current_conflicts: List[Proposal]) -> Optional[Proposal]:
        """If primary drug is vetoed or high conflict, try Carvedilol (beta‑blocker)."""
        if any(p.agent_name == self.name for p in current_conflicts):
            return None  # avoid duplicate
        if patient.vitals.systolic_bp >= 100 and patient.vitals.heart_rate >= 60:
            return Proposal(
                agent_name=self.name,
                treatment="Carvedilol",
                confidence=0.85,
                evidence_level=EvidenceLevel.A,
                risks=["hypotension", "bradycardia"],
                mechanism="beta_blocker",
                alternative_to="Lisinopril/Losartan"
            )
        return None

class NephrologyAgent(BaseClinicalAgent):
    name = "Nephrology"
    def propose(self, patient: Patient) -> Optional[Proposal]:
        if not patient.has_condition("kidney_disease"):
            return None
        # Protect kidneys: recommend ARB for reno‑protection if heart failure present
        if patient.has_condition("heart_failure"):
            return Proposal(
                agent_name=self.name,
                treatment="Losartan",
                confidence=0.88,
                evidence_level=EvidenceLevel.A,
                risks=["hyperkalemia"],
                mechanism="ARB",
                guideline_source="KDIGO 2024",
                guideline_version="2024"
            )
        return None

    def propose_alternative(self, patient: Patient, current_conflicts: List[Proposal]) -> Optional[Proposal]:
        # Could propose Furosemide for volume overload (if appropriate)
        if patient.has_condition("fluid_overload"):
            return Proposal(
                agent_name=self.name,
                treatment="Furosemide",
                confidence=0.80,
                evidence_level=EvidenceLevel.A,
                risks=["hypotension", "electrolyte_imbalance"],
                mechanism="loop_diuretic",
                alternative_to="Losartan"
            )
        return None

class IDAgent(BaseClinicalAgent):
    name = "Infectious Disease"
    def propose(self, patient: Patient) -> Optional[Proposal]:
        if not any("infection" in c for c in patient.conditions):
            return None
        # Select antibiotic based on renal function and allergies
        if patient.labs.egfr < 30:
            # Avoid nephrotoxins -> Ceftriaxone (if no cephalosporin allergy)
            drug = "Ceftriaxone"
        else:
            drug = "Gentamicin"  # more potent but nephrotoxic
        if patient.has_any_allergy_class(["aminoglycoside", "cephalosporin"]):
            # fallback: safe but not in DB – simulate by choosing Ceftriaxone if no cephalosporin allergy
            if "cephalosporin" not in patient.allergies:
                drug = "Ceftriaxone"
            else:
                return None  # cannot treat safely
        info = DRUG_DB[drug]
        return Proposal(
            agent_name=self.name,
            treatment=drug,
            confidence=0.82,
            evidence_level=info.evidence_level,
            risks=list(info.risks.keys()),
            mechanism=info.mechanism,
            guideline_source="IDSA 2024",
            guideline_version="2024"
        )

    def propose_alternative(self, patient: Patient, current_conflicts: List[Proposal]) -> Optional[Proposal]:
        # Alternative: Ceftriaxone if not already proposed
        if not any(p.treatment == "Ceftriaxone" for p in current_conflicts):
            if not patient.has_any_allergy_class(["cephalosporin"]):
                return Proposal(
                    agent_name=self.name,
                    treatment="Ceftriaxone",
                    confidence=0.78,
                    evidence_level=EvidenceLevel.A,
                    risks=["gi_upset", "allergic_reaction"],
                    mechanism="cephalosporin",
                    alternative_to="Gentamicin"
                )
        return None

# ===========================================================================
# 7. CONFLICT ENGINE (IMPROVED MODELING)
# ===========================================================================

@dataclass
class ConflictDetail:
    pair: Tuple[str, str]  # agent/drug names
    drug_interaction_score: float
    organ_overlap_score: float
    cumulative_toxicity_score: float
    total: float

class ConflictEngine:
    @staticmethod
    def _organ_overlap(risk1: str, risk2: str) -> float:
        if risk1 not in RISK_SEVERITY_MAP or risk2 not in RISK_SEVERITY_MAP:
            return 0.0
        sev1, org1 = RISK_SEVERITY_MAP[risk1]
        sev2, org2 = RISK_SEVERITY_MAP[risk2]
        if org1 == org2:
            return 0.8 * (sev1 + sev2) / 2  # same organ, high conflict
        # cross‑organ penalty
        cross = ORGAN_CROSS_PENALTY.get((org1, org2), 0.2)
        return cross * (sev1 + sev2) / 2

    @staticmethod
    def _drug_interaction(p1: Proposal, p2: Proposal) -> float:
        """Check if the two drugs directly interact."""
        info1 = p1.drug_info()
        info2 = p2.drug_info()
        if not info1 or not info2:
            return 0.0
        # Direct interaction?
        if p2.treatment in info1.interactions or p1.treatment in info2.interactions:
            return 0.9  # high penalty
        # If both have same risk (e.g., both cause hypotension) -> increased burden
        common_risks = set(p1.risks) & set(p2.risks)
        if common_risks:
            return 0.4 * len(common_risks)
        return 0.0

    @staticmethod
    def compute_conflict(p1: Proposal, p2: Proposal) -> ConflictDetail:
        # Organ overlap: sum over all risk pairs
        organ_score = 0.0
        for r1 in p1.risks:
            for r2 in p2.risks:
                organ_score += ConflictEngine._organ_overlap(r1, r2)
        # Drug interaction
        interaction_score = ConflictEngine._drug_interaction(p1, p2)
        # Cumulative toxicity: if both agents propose drugs with risk severity > 0.8 to same organ
        cumulative = 0.0
        for r1 in p1.risks:
            for r2 in p2.risks:
                if r1 in RISK_SEVERITY_MAP and r2 in RISK_SEVERITY_MAP:
                    s1, o1 = RISK_SEVERITY_MAP[r1]
                    s2, o2 = RISK_SEVERITY_MAP[r2]
                    if o1 == o2 and s1 > 0.7 and s2 > 0.7:
                        cumulative += 0.7  # additive toxicity
        total = organ_score + interaction_score + cumulative
        return ConflictDetail(
            pair=(f"{p1.agent_name}/{p1.treatment}", f"{p2.agent_name}/{p2.treatment}"),
            drug_interaction_score=round(interaction_score, 3),
            organ_overlap_score=round(organ_score, 3),
            cumulative_toxicity_score=round(cumulative, 3),
            total=round(total, 3)
        )

# ===========================================================================
# 8. SCORING MODEL (CONFIGURABLE)
# ===========================================================================

@dataclass
class ScoringConfig:
    alpha: float = 0.35      # confidence weight
    beta: float = 0.30       # evidence weight
    gamma: float = 0.15      # biological risk weight
    delta: float = 0.10      # interaction penalty weight
    epsilon: float = 0.10    # contraindication penalty weight

class Scorer:
    def __init__(self, config: ScoringConfig = ScoringConfig()):
        self.config = config

    def _evidence_value(self, level: EvidenceLevel) -> float:
        return level.value

    def _biological_risk(self, proposal: Proposal) -> float:
        """Sum of severity values for each risk."""
        total = 0.0
        for risk in proposal.risks:
            sev_info = RISK_SEVERITY_MAP.get(risk)
            if sev_info:
                total += sev_info[0]  # severity score
        return total

    def _interaction_penalty(self, patient: Patient, proposal: Proposal) -> float:
        """Penalty for interactions with patient's current medications (non‑veto interactions)."""
        drug = proposal.drug_info()
        if not drug:
            return 0.0
        penalty = 0.0
        for med in patient.medications:
            if med in drug.interactions:
                desc = drug.interactions[med]
                if "severe" not in desc:  # critical are already vetoed
                    penalty += 0.5
        return penalty

    def _contraindication_penalty(self, patient: Patient, proposal: Proposal) -> float:
        """Penalty for soft contraindications not triggering veto."""
        drug = proposal.drug_info()
        if not drug:
            return 0.0
        pen = 0.0
        for cond, ctype in drug.contraindications.items():
            if ctype == ContraindicationType.ALLERGY:
                continue  # handled by veto
            if "renal_impairment" in cond and patient.labs.egfr < 60:
                pen += 0.4
            if "hypotension" in cond and patient.vitals.systolic_bp < 100:
                pen += 0.3
        return pen

    def compute_score(self, patient: Patient, proposal: Proposal) -> float:
        conf = proposal.confidence
        ev = self._evidence_value(proposal.evidence_level)
        bio_risk = self._biological_risk(proposal)
        int_pen = self._interaction_penalty(patient, proposal)
        contra_pen = self._contraindication_penalty(patient, proposal)

        score = (self.config.alpha * conf
                 + self.config.beta * ev
                 - self.config.gamma * bio_risk
                 - self.config.delta * int_pen
                 - self.config.epsilon * contra_pen)
        return round(score, 4)

# ===========================================================================
# 9. EMERGENCY DETECTOR
# ===========================================================================

class EmergencyDetector:
    @staticmethod
    def detect_sepsis(vitals: VitalSigns, labs: LabResults) -> bool:
        """SIRS criteria (simplified)."""
        criteria = 0
        if vitals.temperature_c > 38.0 or vitals.temperature_c < 36.0:
            criteria += 1
        if vitals.heart_rate > 90:
            criteria += 1
        if vitals.respiratory_rate > 20:
            criteria += 1
        if labs.wbc and (labs.wbc > 12_000 or labs.wbc < 4_000):
            criteria += 1
        return criteria >= 2

    @staticmethod
    def detect_shock(vitals: VitalSigns, labs: LabResults) -> bool:
        """Crude shock detection: hypotension + tachycardia + possible lactate."""
        if vitals.systolic_bp < 90 and vitals.heart_rate > 100:
            if labs.lactate and labs.lactate > 2.0:
                return True
            # even without lactate, strong hypotension with tachycardia is alarming
            return True
        return False

    @classmethod
    def is_emergency(cls, patient: Patient) -> Optional[str]:
        if cls.detect_shock(patient.vitals, patient.labs):
            return "SHOCK"
        if cls.detect_sepsis(patient.vitals, patient.labs):
            return "SEPSIS"
        return None

# ===========================================================================
# 10. ORCHESTRATOR (WITH NEGOTIATION & ESCALATION)
# ===========================================================================

@dataclass
class OrchestrationResult:
    final_recommendation: Optional[Proposal]
    alternatives: List[Proposal]
    vetoed: List[SafetyVeto]
    conflict_details: List[ConflictDetail]
    total_conflict: float
    negotiation_occurred: bool
    emergency_override: Optional[str]
    rationale: str = ""

class Orchestrator:
    def __init__(self, agents: List[BaseClinicalAgent], hca: HCA, scorer: Scorer,
                 conflict_threshold: float = 1.5):
        self.agents = agents
        self.hca = hca
        self.scorer = scorer
        self.conflict_engine = ConflictEngine()
        self.conflict_threshold = conflict_threshold

    def evaluate(self, patient: Patient) -> OrchestrationResult:
        print("\n" + "="*70)
        print("MACO v2.2 ORCHESTRATION REPORT".center(70))
        print("="*70)
        print(f"Patient: {patient.id} | Age: {patient.age} | eGFR: {patient.labs.egfr:.1f}")
        print(f"Conditions: {patient.conditions}")
        print(f"Allergies: {patient.allergies}")
        print(f"Meds: {patient.medications}")
        print("-"*70)

        # Emergency override
        emergency = EmergencyDetector.is_emergency(patient)
        if emergency:
            print(f"[EMERGENCY] {emergency} detected – activating interrupt protocol.")
            # In a real system, bypasses negotiation and returns highest‑priority emergency bundle.
            # For prototype, we return a synthetic recommendation.
            return OrchestrationResult(
                final_recommendation=None,
                alternatives=[],
                vetoed=[],
                conflict_details=[],
                total_conflict=0.0,
                negotiation_occurred=False,
                emergency_override=emergency,
                rationale="Emergency override activated. Immediate stabilisation required."
            )

        # 1. Collect proposals
        proposals: List[Proposal] = []
        for agent in self.agents:
            prop = agent.propose(patient)
            if prop and prop.treatment not in [p.treatment for p in proposals]:  # avoid duplicates
                proposals.append(prop)
        print(f"\n📋 Initial proposals ({len(proposals)}):")
        for p in proposals:
            print(f"  - {p.agent_name}: {p.treatment} (confidence={p.confidence}, evidence={p.evidence_level.name})")

        # 2. HCA validation (deterministic safety)
        valid_proposals: List[Proposal] = []
        vetoes: List[SafetyVeto] = []
        for prop in proposals:
            safe, veto = self.hca.validate(patient, prop)
            if not safe:
                vetoes.append(veto)
                print(f"\n⛔ VETO: {veto.proposal.agent_name}/{veto.proposal.treatment} → {veto.reason}")
            else:
                valid_proposals.append(prop)
                print(f"✅ SAFE: {prop.agent_name}/{prop.treatment}")

        if not valid_proposals:
            print("\nNo safe proposals remain. Escalating to clinician.")
            return OrchestrationResult(
                final_recommendation=None, alternatives=[], vetoed=vetoes,
                conflict_details=[], total_conflict=float('inf'),
                negotiation_occurred=False, emergency_override=None,
                rationale="All proposals vetoed."
            )

        # 3. Conflict matrix
        print("\n" + "-"*70)
        print("CONFLICT MATRIX")
        conflicts = []
        total_conflict = 0.0
        for i in range(len(valid_proposals)):
            for j in range(i+1, len(valid_proposals)):
                detail = self.conflict_engine.compute_conflict(valid_proposals[i], valid_proposals[j])
                conflicts.append(detail)
                total_conflict += detail.total
                print(f"  • {detail.pair[0]} vs {detail.pair[1]}: total={detail.total:.3f} "
                      f"(drug_int={detail.drug_interaction_score:.3f}, organ={detail.organ_overlap_score:.3f}, "
                      f"cum_tox={detail.cumulative_toxicity_score:.3f})")
        print(f"Global conflict score: {total_conflict:.3f}")

        # 4. Negotiation if needed
        negotiation_occurred = False
        final_candidates = valid_proposals
        if total_conflict > self.conflict_threshold:
            negotiation_occurred = True
            print("\n[!] Conflict exceeds threshold → Negotiation loop activated.")
            # Collect alternative proposals
            alternatives: List[Proposal] = []
            for agent in self.agents:
                alt = agent.propose_alternative(patient, valid_proposals)
                if alt and alt.treatment not in [p.treatment for p in valid_proposals]:
                    safe, veto = self.hca.validate(patient, alt)
                    if safe:
                        alternatives.append(alt)
                        print(f"  Alternative from {agent.name}: {alt.treatment}")
            if alternatives:
                # Combine all proposals (original + alternatives) and find best combination
                all_options = valid_proposals + alternatives
                # For simplicity, try all possible subsets (brute force, size up to 4)
                best_combo = valid_proposals  # fallback
                min_conflict = float('inf')
                # We limit to subsets of size len(valid_proposals) to maintain multi‑agent coverage
                from itertools import combinations
                for combo in combinations(all_options, r=len(valid_proposals)):
                    # Ensure unique agents in combo (optional, but we avoid double‑agent)
                    agents_in = set(p.agent_name for p in combo)
                    if len(agents_in) != len(valid_proposals):
                        continue
                    total_c = 0.0
                    for (a,b) in itertools.combinations(combo, 2):
                        detail = self.conflict_engine.compute_conflict(a,b)
                        total_c += detail.total
                    if total_c < min_conflict:
                        min_conflict = total_c
                        best_combo = list(combo)
                if min_conflict < total_conflict:
                    final_candidates = best_combo
                    print(f"  → Optimised conflict: {min_conflict:.3f} (was {total_conflict:.3f})")
                else:
                    print("  → No improvement; keeping original set.")
            else:
                print("  → No alternative proposals available; escalating conflict warning.")

        # 5. Scoring
        print("\n" + "-"*70)
        print("FINAL SCORING")
        scored = [(p, self.scorer.compute_score(patient, p)) for p in final_candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        for prop, score in scored:
            print(f"  {prop.agent_name:15s} | {prop.treatment:15s} | score={score:.4f}")

        # 6. Recommendation
        winner = scored[0][0] if scored else None
        reasons = []
        if vetoes:
            reasons.append(f"{len(vetoes)} veto(s) applied: {', '.join(v.reason for v in vetoes)}")
        if conflicts:
            reasons.append(f"Conflict resolution: {len(conflicts)} pairwise conflicts evaluated")
        if negotiation_occurred:
            reasons.append("Negotiation loop was executed")
        if winner:
            reasons.append(
                f"Selected {winner.treatment} based on agent {winner.agent_name}, "
                f"evidence {winner.evidence_level.name}, guideline {winner.guideline_source} {winner.guideline_version}"
            )
        rationale = "; ".join(reasons)

        print("\n" + "="*70)
        print("FINAL RECOMMENDATION")
        if winner:
            print(f"  Treatment   : {winner.treatment}")
            print(f"  Agent       : {winner.agent_name}")
            print(f"  Evidence    : {winner.evidence_level.name}")
            print(f"  Guideline   : {winner.guideline_source} v{winner.guideline_version}")
            print(f"  Risks       : {', '.join(winner.risks)}")
            print(f"  Rationale   : {rationale}")
            # Print rejected alternatives
            rejected = [f"{p.agent_name}/{p.treatment}" for p,_ in scored if p != winner]
            if rejected:
                print(f"  Rejected    : {', '.join(rejected)}")
        else:
            print("  No safe recommendation could be made.")
        print("="*70 + "\n")

        return OrchestrationResult(
            final_recommendation=winner,
            alternatives=[p for p,_ in scored if p != winner],
            vetoed=vetoes,
            conflict_details=conflicts,
            total_conflict=total_conflict,
            negotiation_occurred=negotiation_occurred,
            emergency_override=emergency,
            rationale=rationale
        )

# ===========================================================================
# 11. SIMULATION / DEMO
# ===========================================================================

if __name__ == "__main__":
    # Build a complex patient
    patient = Patient(
        id="PX-2026",
        age=72,
        weight_kg=68.5,
        conditions=["heart_failure", "kidney_disease", "infection_bacterial"],
        allergies=["penicillin", "ace_inhibitor"],   # allergic to ACEi (Lisinopril)
        medications=["Metformin", "Furosemide"],     # already on diuretic and metformin
        vitals=VitalSigns(
            systolic_bp=105,
            diastolic_bp=68,
            heart_rate=82,
            respiratory_rate=18,
            temperature_c=37.1,
            spo2=96
        ),
        labs=LabResults(
            creatinine=2.8,
            egfr=28.0,          # CKD Stage 4
            potassium=5.1,
            sodium=138,
            wbc=11_500,
            lactate=1.8
        )
    )

    # Instantiate framework components
    agents = [
        CardiologyAgent(),
        NephrologyAgent(),
        IDAgent(),
    ]
    hca = HCA()
    scorer = Scorer(ScoringConfig(
        alpha=0.35, beta=0.30, gamma=0.15, delta=0.10, epsilon=0.10
    ))
    orchestrator = Orchestrator(agents, hca, scorer, conflict_threshold=1.5)

    # Run orchestration
    result = orchestrator.evaluate(patient)

    # --- Additional simulation for emergency scenario ---
    emerg_patient = Patient(
        id="PX-EMERG",
        age=65,
        weight_kg=80,
        conditions=["sepsis", "heart_failure"],
        allergies=[],
        medications=[],
        vitals=VitalSigns(90, 60, 115, 24, 39.2, 92),
        labs=LabResults(1.2, 80, 4.0, 140, wbc=18_000, lactate=4.5)
    )
    print("\n\n>>> EMERGENCY SIMULATION <<<")
    orchestrator.evaluate(emerg_patient)
