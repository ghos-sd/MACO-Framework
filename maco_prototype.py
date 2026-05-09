"""
===========================================================================
MACO v2.2 – Multi-Agent Clinical Orchestration (FHIR-Ready Refactor)
===========================================================================
"Clinical safety emerges from structured conflict between constrained expert domains."

This version refactors the patient and clinical data models to align with
HL7 FHIR R4 Resources. All observations carry LOINC codes and effective
timestamps; conditions are mapped to SNOMED CT. HCA validation now uses
coded concepts, improving semantic precision and auditability.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set, Union
from enum import Enum, auto
from datetime import datetime
import itertools

# =====================================================================
# 1. ENUMS, CODING SYSTEMS & CONSTANTS
# =====================================================================

class Severity(Enum):
    CRITICAL = 5
    HIGH = 4
    MODERATE = 3
    LOW = 2
    NEGLIGIBLE = 1

class EvidenceLevel(Enum):
    A = 1.0
    B = 0.7
    C = 0.4
    D = 0.2

class ContraindicationType(Enum):
    ALLERGY = auto()
    ORGAN_FAILURE = auto()
    DRUG_INTERACTION = auto()
    HEMODYNAMIC_INSTABILITY = auto()
    LAB_THRESHOLD = auto()

class OrganSystem(Enum):
    CARDIOVASCULAR = "cardiovascular"
    RENAL = "renal"
    HEPATIC = "hepatic"
    RESPIRATORY = "respiratory"
    NEUROLOGICAL = "neurological"
    METABOLIC = "metabolic"
    IMMUNE = "immune"
    GI = "gastrointestinal"

# ---------------------------------------------------------------
# LOINC codes for observations (FHIR R4 Observation.code)
# ---------------------------------------------------------------
LOINC_VITALS = {
    "systolic_bp":     {"code": "8480-6", "display": "Systolic blood pressure", "unit": "mm[Hg]"},
    "diastolic_bp":    {"code": "8462-4", "display": "Diastolic blood pressure", "unit": "mm[Hg]"},
    "heart_rate":      {"code": "8867-4", "display": "Heart rate", "unit": "/min"},
    "respiratory_rate": {"code": "9279-1", "display": "Respiratory rate", "unit": "/min"},
    "temperature_c":   {"code": "8310-5", "display": "Body temperature", "unit": "Cel"},
    "spo2":            {"code": "2708-6", "display": "Oxygen saturation in Arterial blood", "unit": "%"},
}

LOINC_LABS = {
    "creatinine": {"code": "2160-0", "display": "Creatinine [Mass/volume] in Serum or Plasma", "unit": "mg/dL"},
    "egfr":       {"code": "62238-1", "display": "Glomerular filtration rate/1.73 sq M predicted", "unit": "mL/min/1.73m2"},
    "potassium":  {"code": "2823-3", "display": "Potassium [Moles/volume] in Serum or Plasma", "unit": "mmol/L"},
    "sodium":     {"code": "2951-2", "display": "Sodium [Moles/volume] in Serum or Plasma", "unit": "mmol/L"},
    "wbc":        {"code": "26464-8", "display": "Leukocytes [#/volume] in Blood", "unit": "/mm3"},
    "lactate":    {"code": "2524-6", "display": "Lactate [Moles/volume] in Blood", "unit": "mmol/L"},
}

# ---------------------------------------------------------------
# SNOMED CT codes for conditions (FHIR Condition.code)
# ---------------------------------------------------------------
SNOMED_CONDITIONS = {
    # internal key -> (SNOMED code, display)
    "heart_failure":            ("84114007", "Heart failure (disorder)"),
    "kidney_disease":           ("709044004", "Chronic kidney disease (disorder)"),
    "hypertension":             ("38341003", "Hypertensive disorder, systemic arterial (disorder)"),
    "renal_impairment_severe":  ("431855005", "Chronic kidney disease stage 4 (disorder)"),
    "fluid_overload":           ("42399005", "Fluid overload (disorder)"),
    "infection_bacterial":      ("40733004", "Bacterial infection (disorder)"),
    "sepsis":                   ("91302008", "Sepsis (disorder)"),
}

# Map from old string keys to SNOMED display; used for backward‑compatible
# patient.has_condition() matching.
_SNOMED_STR_TO_DISPLAY = {k: v[1] for k, v in SNOMED_CONDITIONS.items()}

# ---------------------------------------------------------------
# Allergy codes (simplified, could be extended to SNOMED substances)
# ---------------------------------------------------------------
# In a full FHIR model, allergies would be AllergyIntolerance resources
# with a code from a substance hierarchy. We keep the string `allergy_class`
# as the primary matching mechanism, but add a mapping to SNOMED for reference.
ALLERGY_CLASS_SNOMED = {
    "penicillin":    "373270004",    # Penicillin (substance)
    "sulfonamide":   "387406002",    # Sulfonamide (substance)
    "ace_inhibitor": "96352001",     # Angiotensin‑converting enzyme inhibitor (substance)
    "aminoglycoside":"360204005",    # Aminoglycoside (substance)
    "cephalosporin": "373186003",    # Cephalosporin (substance)
}

# Risk severity map (unchanged)
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

ORGAN_CROSS_PENALTY: Dict[Tuple[OrganSystem, OrganSystem], float] = {
    (OrganSystem.RENAL, OrganSystem.CARDIOVASCULAR): 0.6,
    (OrganSystem.CARDIOVASCULAR, OrganSystem.RENAL): 0.6,
    (OrganSystem.HEPATIC, OrganSystem.RENAL): 0.5,
    (OrganSystem.RENAL, OrganSystem.HEPATIC): 0.5,
    (OrganSystem.CARDIOVASCULAR, OrganSystem.RESPIRATORY): 0.5,
    (OrganSystem.RESPIRATORY, OrganSystem.CARDIOVASCULAR): 0.5,
}

# =====================================================================
# 2. DRUG DATABASE (unchanged except for coded condition keys)
# =====================================================================

@dataclass
class DrugInfo:
    name: str
    indications: List[str]
    contraindications: Dict[str, ContraindicationType]  # condition keys (internal strings)
    risks: Dict[str, float]
    interactions: Dict[str, str]
    evidence_level: EvidenceLevel
    mechanism: str
    guideline_source: str = "NICE"
    guideline_version: str = "2024"
    allergy_class: Optional[str] = None

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

ALLERGY_DRUG_MAP: Dict[str, Set[str]] = {
    "penicillin": {"Penicillin_V", "Amoxicillin", "Ampicillin"},
    "sulfonamide": {"Sulfamethoxazole"},
    "ace_inhibitor": {"Lisinopril", "Enalapril"},
    "aminoglycoside": {"Gentamicin", "Tobramycin"},
    "cephalosporin": {"Ceftriaxone", "Cefazolin"},
}

# =====================================================================
# 3. FHIR-ALIGNED CLINICAL DATA MODELS
# =====================================================================

@dataclass
class Condition:
    """FHIR Condition resource (simplified)."""
    system: str = "http://snomed.info/sct"
    code: str = ""
    display: str = ""
    # Optional clinical status and verification status can be added if needed.

    def __eq__(self, other) -> bool:
        if isinstance(other, Condition):
            return self.code == other.code
        return False

    def __hash__(self) -> int:
        return hash(self.code)

@dataclass
class VitalSigns:
    """Vital signs as FHIR Observations. Each attribute corresponds to a LOINC code."""
    systolic_bp: float          # mm[Hg]
    diastolic_bp: float
    heart_rate: float           # /min
    respiratory_rate: float
    temperature_c: float        # Celsius
    spo2: float                 # %
    effective_datetime: Optional[datetime] = None  # FHIR effectiveDateTime (R4)

    # FHIR mapping: internal attribute -> LOINC coding
    FHIR_MAPPING = LOINC_VITALS

    def to_fhir_observations(self) -> List[Dict[str, Any]]:
        """Generate a list of simplified FHIR Observation dictionaries with code and value."""
        obs_list = []
        for attr, loinc in self.FHIR_MAPPING.items():
            value = getattr(self, attr)
            if value is None:
                continue
            obs = {
                "resourceType": "Observation",
                "status": "final",
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": loinc["code"],
                        "display": loinc["display"]
                    }]
                },
                "valueQuantity": {
                    "value": value,
                    "unit": loinc["unit"],
                    "system": "http://unitsofmeasure.org",
                    "code": loinc["unit"]
                },
                "effectiveDateTime": self.effective_datetime.isoformat() if self.effective_datetime else None
            }
            obs_list.append(obs)
        return obs_list

@dataclass
class LabResults:
    """Lab results as FHIR Observations."""
    creatinine: float           # mg/dL
    egfr: float                 # mL/min/1.73m²
    potassium: float            # mmol/L
    sodium: float
    wbc: Optional[float] = None # /mm3
    lactate: Optional[float] = None  # mmol/L
    effective_datetime: Optional[datetime] = None

    FHIR_MAPPING = LOINC_LABS

    def to_fhir_observations(self) -> List[Dict[str, Any]]:
        obs_list = []
        for attr, loinc in self.FHIR_MAPPING.items():
            value = getattr(self, attr)
            if value is None:
                continue
            obs = {
                "resourceType": "Observation",
                "status": "final",
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": loinc["code"],
                        "display": loinc["display"]
                    }]
                },
                "valueQuantity": {
                    "value": value,
                    "unit": loinc["unit"],
                    "system": "http://unitsofmeasure.org",
                    "code": loinc["unit"]
                },
                "effectiveDateTime": self.effective_datetime.isoformat() if self.effective_datetime else None
            }
            obs_list.append(obs)
        return obs_list

@dataclass
class Patient:
    id: str
    age: int
    weight_kg: float
    conditions: List[Condition]                     # SNOMED CT coded conditions
    allergies: List[str]                            # keep as strings for now (could be extended to Substance codes)
    medications: List[str]
    vitals: VitalSigns
    labs: LabResults

    def has_condition(self, identifier: str) -> bool:
        """
        Check if patient has a condition matching the given identifier.
        The identifier can be a SNOMED code (e.g., '84114007'), a display name
        (e.g., 'Heart failure'), or an old internal string key (e.g., 'heart_failure').
        """
        # First map old string keys to SNOMED display if possible
        mapped_display = _SNOMED_STR_TO_DISPLAY.get(identifier, identifier)

        for cond in self.conditions:
            if cond.code == identifier or cond.display.lower() == mapped_display.lower():
                return True
        return False

    def has_condition_code(self, snomed_code: str) -> bool:
        """Check for a specific SNOMED CT code."""
        return any(c.code == snomed_code for c in self.conditions)

    def has_any_allergy_class(self, classes: List[str]) -> bool:
        """
        Check if patient has an allergy that cross‑reacts with any of the given drug classes.
        Still uses string/substring matching; the allergy list may include SNOMED display names.
        In a full FHIR implementation, allergies would be coded as AllergyIntolerance resources.
        """
        for allergy in self.allergies:
            for cls in classes:
                if allergy.lower() in cls.lower() or cls.lower() in allergy.lower():
                    return True
        return False

    @property
    def renal_stage(self) -> str:
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

# Helper to build a Condition from an old string key
def condition_from_key(key: str) -> Condition:
    snomed = SNOMED_CONDITIONS.get(key, ("", key))
    return Condition(system="http://snomed.info/sct", code=snomed[0], display=snomed[1])

# =====================================================================
# 4. PROPOSAL & GUIDELINE (unchanged)
# =====================================================================

@dataclass
class Proposal:
    agent_name: str
    treatment: str
    confidence: float
    evidence_level: EvidenceLevel
    risks: List[str]
    mechanism: str
    guideline_source: str = "NICE"
    guideline_version: str = "2024"
    alternative_to: Optional[str] = None

    def drug_info(self) -> Optional[DrugInfo]:
        return DRUG_DB.get(self.treatment)

# =====================================================================
# 5. SAFETY LAYER – HCA UPDATED TO USE FHIR CODES
# =====================================================================

@dataclass
class SafetyVeto:
    proposal: Proposal
    reason: str
    severity: Severity
    veto_type: ContraindicationType

class HCA:
    """Historical Context Agent – now uses SNOMED/LOINC codes where relevant."""

    @staticmethod
    def _check_allergy(patient: Patient, proposal: Proposal) -> Optional[SafetyVeto]:
        drug = proposal.drug_info()
        if not drug or not drug.allergy_class:
            return None
        # Allergy matching uses the allergy_class string; we can also check SNOMED
        # if allergies were coded, but here we keep the existing logic.
        for allergy in patient.allergies:
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
        contraindications = drug.contraindications

        # Renal impairment check — uses LOINC‑coded eGFR value, condition check via SNOMED
        if "renal_impairment_severe" in contraindications:
            # Check SNOMED code 431855005 (CKD Stage 4) or eGFR threshold
            if patient.labs.egfr < 30 or patient.has_condition_code("431855005"):
                return SafetyVeto(
                    proposal,
                    f"Renal contraindication: eGFR {patient.labs.egfr} or coded CKD Stage 4 with {drug.name}",
                    Severity.CRITICAL,
                    ContraindicationType.ORGAN_FAILURE
                )
        # Hemodynamic instability
        if "severe_hypotension" in contraindications or "bradycardia_severe" in contraindications:
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
                if "severe" in interaction_desc or "life" in interaction_desc:
                    return SafetyVeto(
                        proposal,
                        f"Dangerous interaction: {drug.name} + {med} → {interaction_desc}",
                        Severity.CRITICAL,
                        ContraindicationType.DRUG_INTERACTION
                    )
        return None

    def validate(self, patient: Patient, proposal: Proposal) -> Tuple[bool, Optional[SafetyVeto]]:
        for check in [self._check_allergy, self._check_organ_contraindications, self._check_drug_interactions]:
            veto = check(patient, proposal)
            if veto:
                return False, veto
        return True, None

# =====================================================================
# 6. CLINICAL DOMAIN AGENTS (unchanged except condition checks now handle coded conditions)
# =====================================================================

class BaseClinicalAgent:
    name: str = "BaseAgent"

    def propose(self, patient: Patient) -> Optional[Proposal]:
        raise NotImplementedError

    def propose_alternative(self, patient: Patient, current_conflicts: List[Proposal]) -> Optional[Proposal]:
        return None

class CardiologyAgent(BaseClinicalAgent):
    name = "Cardiology"
    def propose(self, patient: Patient) -> Optional[Proposal]:
        if not patient.has_condition("heart_failure"):
            return None
        if patient.labs.egfr < 30:
            drug = "Losartan"
        else:
            drug = "Lisinopril"
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
        if any(p.agent_name == self.name for p in current_conflicts):
            return None
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
        if not any("infection" in c.display.lower() for c in patient.conditions):  # uses display name
            return None
        if patient.labs.egfr < 30:
            drug = "Ceftriaxone"
        else:
            drug = "Gentamicin"
        if patient.has_any_allergy_class(["aminoglycoside", "cephalosporin"]):
            if "cephalosporin" not in patient.allergies:
                drug = "Ceftriaxone"
            else:
                return None
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

# =====================================================================
# 7. CONFLICT ENGINE (unchanged)
# =====================================================================

@dataclass
class ConflictDetail:
    pair: Tuple[str, str]
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
            return 0.8 * (sev1 + sev2) / 2
        cross = ORGAN_CROSS_PENALTY.get((org1, org2), 0.2)
        return cross * (sev1 + sev2) / 2

    @staticmethod
    def _drug_interaction(p1: Proposal, p2: Proposal) -> float:
        info1 = p1.drug_info()
        info2 = p2.drug_info()
        if not info1 or not info2:
            return 0.0
        if p2.treatment in info1.interactions or p1.treatment in info2.interactions:
            return 0.9
        common_risks = set(p1.risks) & set(p2.risks)
        if common_risks:
            return 0.4 * len(common_risks)
        return 0.0

    @staticmethod
    def compute_conflict(p1: Proposal, p2: Proposal) -> ConflictDetail:
        organ_score = 0.0
        for r1 in p1.risks:
            for r2 in p2.risks:
                organ_score += ConflictEngine._organ_overlap(r1, r2)
        interaction_score = ConflictEngine._drug_interaction(p1, p2)
        cumulative = 0.0
        for r1 in p1.risks:
            for r2 in p2.risks:
                if r1 in RISK_SEVERITY_MAP and r2 in RISK_SEVERITY_MAP:
                    s1, o1 = RISK_SEVERITY_MAP[r1]
                    s2, o2 = RISK_SEVERITY_MAP[r2]
                    if o1 == o2 and s1 > 0.7 and s2 > 0.7:
                        cumulative += 0.7
        total = organ_score + interaction_score + cumulative
        return ConflictDetail(
            pair=(f"{p1.agent_name}/{p1.treatment}", f"{p2.agent_name}/{p2.treatment}"),
            drug_interaction_score=round(interaction_score, 3),
            organ_overlap_score=round(organ_score, 3),
            cumulative_toxicity_score=round(cumulative, 3),
            total=round(total, 3)
        )

# =====================================================================
# 8. SCORING MODEL (unchanged)
# =====================================================================

@dataclass
class ScoringConfig:
    alpha: float = 0.35
    beta: float = 0.30
    gamma: float = 0.15
    delta: float = 0.10
    epsilon: float = 0.10

class Scorer:
    def __init__(self, config: ScoringConfig = ScoringConfig()):
        self.config = config

    def _evidence_value(self, level: EvidenceLevel) -> float:
        return level.value

    def _biological_risk(self, proposal: Proposal) -> float:
        total = 0.0
        for risk in proposal.risks:
            sev_info = RISK_SEVERITY_MAP.get(risk)
            if sev_info:
                total += sev_info[0]
        return total

    def _interaction_penalty(self, patient: Patient, proposal: Proposal) -> float:
        drug = proposal.drug_info()
        if not drug:
            return 0.0
        penalty = 0.0
        for med in patient.medications:
            if med in drug.interactions:
                desc = drug.interactions[med]
                if "severe" not in desc:
                    penalty += 0.5
        return penalty

    def _contraindication_penalty(self, patient: Patient, proposal: Proposal) -> float:
        drug = proposal.drug_info()
        if not drug:
            return 0.0
        pen = 0.0
        for cond, ctype in drug.contraindications.items():
            if ctype == ContraindicationType.ALLERGY:
                continue
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

# =====================================================================
# 9. EMERGENCY DETECTOR (unchanged, but now uses FHIR‑stamped observations)
# =====================================================================

class EmergencyDetector:
    @staticmethod
    def detect_sepsis(vitals: VitalSigns, labs: LabResults) -> bool:
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
        if vitals.systolic_bp < 90 and vitals.heart_rate > 100:
            if labs.lactate and labs.lactate > 2.0:
                return True
            return True
        return False

    @classmethod
    def is_emergency(cls, patient: Patient) -> Optional[str]:
        if cls.detect_shock(patient.vitals, patient.labs):
            return "SHOCK"
        if cls.detect_sepsis(patient.vitals, patient.labs):
            return "SEPSIS"
        return None

# =====================================================================
# 10. ORCHESTRATOR (unchanged)
# =====================================================================

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
        print("MACO v2.2 (FHIR-Ready) ORCHESTRATION REPORT".center(70))
        print("="*70)
        print(f"Patient: {patient.id} | Age: {patient.age} | eGFR: {patient.labs.egfr:.1f}")
        print(f"Conditions: {[c.display for c in patient.conditions]}")
        print(f"Allergies: {patient.allergies}")
        print(f"Meds: {patient.medications}")
        print(f"Vitals timestamp: {patient.vitals.effective_datetime}")
        print(f"Labs timestamp: {patient.labs.effective_datetime}")
        print("-"*70)

        emergency = EmergencyDetector.is_emergency(patient)
        if emergency:
            print(f"[EMERGENCY] {emergency} detected – activating interrupt protocol.")
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

        proposals: List[Proposal] = []
        for agent in self.agents:
            prop = agent.propose(patient)
            if prop and prop.treatment not in [p.treatment for p in proposals]:
                proposals.append(prop)
        print(f"\n📋 Initial proposals ({len(proposals)}):")
        for p in proposals:
            print(f"  - {p.agent_name}: {p.treatment} (confidence={p.confidence}, evidence={p.evidence_level.name})")

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

        negotiation_occurred = False
        final_candidates = valid_proposals
        if total_conflict > self.conflict_threshold:
            negotiation_occurred = True
            print("\n[!] Conflict exceeds threshold → Negotiation loop activated.")
            alternatives: List[Proposal] = []
            for agent in self.agents:
                alt = agent.propose_alternative(patient, valid_proposals)
                if alt and alt.treatment not in [p.treatment for p in valid_proposals]:
                    safe, veto = self.hca.validate(patient, alt)
                    if safe:
                        alternatives.append(alt)
                        print(f"  Alternative from {agent.name}: {alt.treatment}")
            if alternatives:
                all_options = valid_proposals + alternatives
                best_combo = valid_proposals
                min_conflict = float('inf')
                from itertools import combinations
                for combo in combinations(all_options, r=len(valid_proposals)):
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

        print("\n" + "-"*70)
        print("FINAL SCORING")
        scored = [(p, self.scorer.compute_score(patient, p)) for p in final_candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        for prop, score in scored:
            print(f"  {prop.agent_name:15s} | {prop.treatment:15s} | score={score:.4f}")

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

# =====================================================================
# 11. SIMULATION / DEMO (updated with FHIR timestamps and coded conditions)
# =====================================================================

if __name__ == "__main__":
    now = datetime.now()

    patient = Patient(
        id="PX-2026",
        age=72,
        weight_kg=68.5,
        conditions=[
            condition_from_key("heart_failure"),
            condition_from_key("kidney_disease"),
            condition_from_key("infection_bacterial")
        ],
        allergies=["penicillin", "ace_inhibitor"],
        medications=["Metformin", "Furosemide"],
        vitals=VitalSigns(
            systolic_bp=105, diastolic_bp=68, heart_rate=82,
            respiratory_rate=18, temperature_c=37.1, spo2=96,
            effective_datetime=now
        ),
        labs=LabResults(
            creatinine=2.8, egfr=28.0, potassium=5.1, sodium=138,
            wbc=11_500, lactate=1.8,
            effective_datetime=now
        )
    )

    agents = [CardiologyAgent(), NephrologyAgent(), IDAgent()]
    hca = HCA()
    scorer = Scorer(ScoringConfig(alpha=0.35, beta=0.30, gamma=0.15, delta=0.10, epsilon=0.10))
    orchestrator = Orchestrator(agents, hca, scorer, conflict_threshold=1.5)

    result = orchestrator.evaluate(patient)

    # Demonstrate FHIR observation export
    print("----- FHIR Observation Export (first 3 vitals) -----")
    for obs in patient.vitals.to_fhir_observations()[:3]:
        print(obs)

    # Emergency simulation
    emerg_patient = Patient(
        id="PX-EMERG",
        age=65,
        weight_kg=80,
        conditions=[condition_from_key("sepsis"), condition_from_key("heart_failure")],
        allergies=[],
        medications=[],
        vitals=VitalSigns(90, 60, 115, 24, 39.2, 92, effective_datetime=now),
        labs=LabResults(1.2, 80, 4.0, 140, wbc=18_000, lactate=4.5, effective_datetime=now)
    )
    print("\n\n>>> EMERGENCY SIMULATION <<<")
    orchestrator.evaluate(emerg_patient)
