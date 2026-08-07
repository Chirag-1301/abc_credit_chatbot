import os
import re
import time
import json
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
PUBLIC_DIR = os.path.join(BASE_DIR, 'public')
AUDIT_LOG_FILE = os.path.join(LOGS_DIR, 'audit_repository.jsonl')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PUBLIC_DIR, exist_ok=True)

# Load Model & Metadata
model_path = os.path.join(MODEL_DIR, 'lgbm_model.joblib')
metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')

if not os.path.exists(model_path) or not os.path.exists(metadata_path):
    raise RuntimeError("Model files not found. Run ABC_Credit_Loan_Approval_Chatbot_Pipeline.ipynb first.")

model = joblib.load(model_path)
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

pincode_risk_dict = metadata['pincode_risk_dict']
make_risk_dict = metadata['make_risk_dict']
pin_to_score = metadata.get('pin_to_score', {})
make_to_cc = metadata.get('make_to_cc', {})
global_mean = metadata['global_mean_decline']
overall_pros = float(metadata.get('overall_prosperity_score', 50.0))
# Operational Credit Risk Threshold
# Model benchmark optimal threshold = 0.0990 (9.9% PD).
# Operating threshold set to 0.030 (3.0% PD) for strict, conservative underwriting.
OPERATIONAL_RISK_THRESHOLD = 0.030
decision_threshold = OPERATIONAL_RISK_THRESHOLD
feature_cols = metadata['feature_cols']
cat_cols = metadata['cat_cols']

app = FastAPI(title="ABC Credit Binary Loan Decision API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PII Masking Utilities
def mask_name(name: Optional[str]) -> str:
    if not name or len(name.strip()) == 0:
        return "A*** A***"
    parts = name.strip().split()
    masked_parts = []
    for p in parts:
        if len(p) <= 2:
            masked_parts.append(p[0] + "*")
        else:
            masked_parts.append(p[0] + "*" * (len(p) - 1))
    return " ".join(masked_parts)

def mask_phone(phone: Optional[str]) -> str:
    if not phone:
        return "+91 ***** **000"
    digits = ''.join(filter(str.isdigit, str(phone)))
    if len(digits) >= 10:
        return f"+91 ***** **{digits[-3:]}"
    return "+91 ***** **" + digits[-2:] if len(digits) >= 2 else "+91 ***** ***"

def mask_email(email: Optional[str]) -> str:
    if not email or "@" not in email:
        return "a***@e******.com"
    username, domain = email.strip().split("@", 1)
    masked_user = username[0] + "***" + (username[-1] if len(username) > 1 else "")
    dom_parts = domain.split(".")
    masked_dom = dom_parts[0][0] + "******" + ("." + dom_parts[1] if len(dom_parts) > 1 else ".com")
    return f"{masked_user}@{masked_dom}"

def mask_pincode(pincode: Any) -> str:
    pin_str = str(pincode).split(".")[0].zfill(6)
    if len(pin_str) >= 6:
        return f"{pin_str[:2]}****"
    return pin_str[:1] + "***"

# Groq LLM Initialization
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY, timeout=5.0)
        print("Groq LLM Client initialized successfully (llama-3.1-8b-instant).")
    except Exception as e:
        print(f"Failed to initialize Groq client: {e}")

class ChatTurnRequest(BaseModel):
    session_id: Optional[str] = None
    user_message: str
    collected_facts: Dict[str, Any] = {}

LEARNED_VEHICLES_FILE = os.path.join(DATA_DIR, 'dynamically_learned_vehicles.json')

def load_learned_vehicles() -> Dict[str, Any]:
    if os.path.exists(LEARNED_VEHICLES_FILE):
        try:
            with open(LEARNED_VEHICLES_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_learned_vehicle(raw_name: str, details: Dict[str, Any]):
    learned = load_learned_vehicles()
    learned[raw_name.lower().strip()] = details
    try:
        with open(LEARNED_VEHICLES_FILE, 'w') as f:
            json.dump(learned, f, indent=2)
        print(f"💡 Dynamically learned and saved new vehicle on-the-fly: '{raw_name}' -> {details}")
    except Exception as e:
        print(f"Error saving learned vehicle: {e}")

def lookup_unseen_vehicle_on_the_fly(vehicle_name: str) -> Optional[Dict[str, Any]]:
    """Queries Groq LLM to check if vehicle_name is a valid motorcycle/scooter model and retrieve technical specs."""
    vehicle_name = vehicle_name.strip()
    if len(vehicle_name) < 3 or vehicle_name.lower() in ['hi', 'hello', 'chirag', 'rahul', 'test', 'yes', 'no', 'male', 'female']:
        return None

    key = vehicle_name.lower()
    learned_cache = load_learned_vehicles()
    if key in learned_cache:
        return learned_cache[key]
        
    if groq_client:
        prompt = (
            f"Determine if '{vehicle_name}' is a real two-wheeler motorcycle or scooter model/brand.\n"
            f"Return JSON:\n"
            f"- is_valid_vehicle (boolean: true if it is a real motorcycle/scooter/brand like Hayabusa, Gixxer, Jupiter, Activa, Splendor, Pulsar, Bullet, KTM, Ninja, Vespa. false if it is a person name, greeting, city, or non-vehicle word).\n"
            f"- canonical_name (string)\n"
            f"- engine_cc (number)\n"
            f"- baseline_onroad_price (number in INR)\n"
            f"- mapped_make_code (MUST be one of dataset categories: JUPITER, ACTIVA, SPLENDOR, APACHE, PULSAR, BULLET, ACCESS, SHINE, XL100)\n"
            f"Output ONLY valid raw JSON."
        )
        try:
            res = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.0,
                max_tokens=150
            )
            content = res.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                data = json.loads(content)
                
            if not data.get("is_valid_vehicle", False):
                return None
                
            learned_details = {
                "canonical_name": data.get("canonical_name", vehicle_name.title()),
                "engine_cc": float(data.get("engine_cc", 125.0)),
                "baseline_price": float(data.get("baseline_onroad_price", 110000.0)),
                "make_code": data.get("mapped_make_code", "APACHE"),
                "learned_at": datetime.now().isoformat()
            }
            save_learned_vehicle(key, learned_details)
            return learned_details
        except Exception as e:
            print(f"On-the-fly Groq vehicle lookup error: {e}")
            
    return None

def get_catalog_price_for_make(make_code: str, vehicle_name: str = None) -> float:
    """Dynamically fetches expected on-road vehicle price using Groq LLM on-the-fly or catalog CSV."""
    query_name = vehicle_name or make_code
    if query_name:
        learned_info = lookup_unseen_vehicle_on_the_fly(query_name)
        if learned_info and 'baseline_price' in learned_info:
            return float(learned_info['baseline_price'])
            
    if os.path.exists(VEHICLE_CATALOG_FILE):
        try:
            df_cat = pd.read_csv(VEHICLE_CATALOG_FILE)
            if vehicle_name:
                v_match = df_cat[df_cat['Model_Description'].str.lower() == vehicle_name.lower().strip()]
                if not v_match.empty:
                    return float(v_match.iloc[0]['Typical_Price'])
            if make_code:
                m_match = df_cat[df_cat['Make_Code'].str.upper() == str(make_code).upper().strip()]
                if not m_match.empty:
                    return float(m_match.iloc[0]['Typical_Price'])
        except Exception:
            pass

    return 110000.0

def extract_facts_with_groq(text: str, current_facts: Dict[str, Any], pending_feature: str = None) -> Dict[str, Any]:
    """Uses Groq LLM to parse raw user natural language directly into structured JSON slots without regex pre-processing."""
    if not groq_client:
        return extract_facts_from_text(text, current_facts)
        
    # STRICT SLOT MAPPING: Only extract exactly the pending_feature slot from user input.
    # This prevents cross-slot contamination — e.g. if pending_feature='net_salary', only extract net_salary.
    # If pending_feature='make_code', only extract make_code/vehicle_raw_name — never interpret as person name.
    
    slot_instruction = ""
    if pending_feature == 'make_code':
        slot_instruction = (
            "STRICT TASK: The user is answering the question 'Which two-wheeler model are you purchasing?'."
            " Extract ONLY: vehicle_raw_name (exact name if it is a two-wheeler motorcycle/scooter brand/model like Gixxer, Pulsar, Activa, Bullet, KTM, Jupiter, Splendor, Apache, Shine, Access, XL100, Hayabusa, Ninja, RE, TVS, Honda, Hero, Suzuki, Yamaha)."  
            " Also set make_code if it maps to a known dataset code (JUPITER, ACTIVA, SPLENDOR, APACHE, PULSAR, BULLET, ACCESS, SHINE, XL100)."  
            " If the user input does NOT look like a motorcycle/scooter model, set is_digression: true and ask them to specify which two-wheeler they are buying."
        )
    elif pending_feature == 'vehicle_price':
        slot_instruction = (
            "STRICT TASK: The user is answering 'What is your negotiated on-road deal price for the vehicle (in ₹)?'."
            " Extract ONLY: vehicle_price as a number in INR. Convert shorthands: '1.1 lakh' -> 110000, '85k' -> 85000."
            " Do NOT extract any other slot."
        )
    elif pending_feature == 'loan_amount':
        slot_instruction = (
            "STRICT TASK: The user is answering 'What loan amount do you need (in ₹)?'."
            " Extract ONLY: loan_amount as a number in INR. Convert shorthands: '85k' -> 85000, '1 lakh' -> 100000."
            " Do NOT extract any other slot."
        )
    elif pending_feature == 'net_salary':
        slot_instruction = (
            "STRICT TASK: The user is answering 'What is your approximate net monthly salary/income (in ₹)?'."
            " Extract ONLY: net_salary as a number in INR. Convert shorthands: '30k' -> 30000, '45k' -> 45000."
            " Do NOT extract any other slot."
        )
    elif pending_feature == 'employment_type':
        slot_instruction = (
            "STRICT TASK: The user is answering 'What is your employment sector?'."
            " Extract ONLY: employment_type as one of: SAL (salaried/job/employee), SEP (self-employed professional/doctor/CA/lawyer/consultant), "
            "AGR (farmer/agriculture/kisan), NREGI (shopkeeper/trader/store owner), STU (student/college), "
            "NPP (freelancer/contractor/gig worker/private work/independent), PEN (retired/pensioner), NONEARNMEM (homemaker/housewife/non-earning)."
            " Do NOT extract any other slot."
        )
    elif pending_feature == 'resident_type':
        slot_instruction = (
            "STRICT TASK: The user is answering 'What is your residential status?'."
            " Extract ONLY: resident_type as one of: O (owned/self-owned/own house), R (rented/rent), L (leased), CO (company provided/accommodation by company)."
            " Do NOT extract any other slot."
        )
    elif pending_feature == 'pincode':
        slot_instruction = (
            "STRICT TASK: The user is answering 'What is your residential pincode?'."
            " Extract ONLY: pincode as a 6-digit numeric string starting with 1-8."
            " If the value is not a valid 6-digit pincode, set pincode: null and validation_error: 'Please enter a valid 6-digit Pincode.'."
            " Do NOT extract any other slot."
        )
    elif pending_feature == 'age':
        slot_instruction = (
            "STRICT TASK: The user is answering 'What is your age?'."
            " Extract ONLY: age as an integer between 18 and 70."
            " If outside range, set age: null and validation_error: 'Applicant age must be between 18 and 70 years.'."
            " Do NOT extract any other slot."
        )
    else:
        slot_instruction = (
            "Extract all relevant facts from the user message."
            " Check for is_digression if user is asking unrelated questions."
        )

    system_prompt = (
        "You are ABC Credit's instant AI loan approval assistant NLU parser.\n"
        "\n"
        + slot_instruction + "\n"
        "\n"
        "CONVERSION RULES (apply when extracting numbers):\n"
        "- '30k' or '30 K' -> 30000, '45k' -> 45000, '85k' -> 85000, '1.2 lakh' -> 120000, '1.5L' -> 150000.\n"
        "\n"
        "STRICT ZERO-HALLUCINATION RULES:\n"
        "- NEVER answer off-topic questions, interest rates, weather, external queries, or general knowledge.\n"
        "- DO NOT state, invent, or speculate any interest rates or unverified facts!\n"
        "- If user asks an off-topic question, set 'is_digression': true.\n"
        "\n"
        "Output ONLY raw valid JSON, no markdown, no explanation."
    )

    user_prompt_content = f"Existing facts collected: {json.dumps(current_facts)}."
    if pending_feature:
        user_prompt_content += f" Pending question slot to fill: '{pending_feature}'."
    user_prompt_content += f" User raw message: '{text}'"
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_content}
            ],
            temperature=0.0,
            max_tokens=200
        )
        content = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            extracted = json.loads(json_match.group(0))
        else:
            extracted = json.loads(content)
        
        merged = dict(current_facts)
        for k, v in extracted.items():
            if k not in ['is_digression', 'validation_error'] and v is not None and v != "":
                merged[k] = v
                
        if extracted.get('is_digression'):
            merged['_is_digression'] = True
        else:
            merged.pop('_is_digression', None)
            
        if extracted.get('validation_error'):
            merged['_validation_error'] = extracted['validation_error']
        else:
            merged.pop('_validation_error', None)
                
        # On-the-Fly Learning for Unseen Vehicles
        if 'vehicle_raw_name' in extracted and 'make_code' not in current_facts:
            raw_v = extracted['vehicle_raw_name']
            learned_info = lookup_unseen_vehicle_on_the_fly(raw_v)
            if learned_info:
                merged['make_code'] = learned_info['make_code']
                merged['vehicle_name'] = learned_info['canonical_name']
                merged['suggested_price'] = learned_info['baseline_price']
                
        return merged
    except Exception as e:
        print(f"Groq LLM extraction fallback: {e}")
        return extract_facts_from_text(text, current_facts, pending_feature)

def normalize_user_text(text: str) -> str:
    """Normalizes raw user input text converting currency shorthands like '110k' -> '110000', '1.1 lakh' -> '110000'."""
    t = text.strip().lower().replace(',', '')
    t = re.sub(r'(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|l)\b', lambda m: str(int(float(m.group(1)) * 100000)), t)
    t = re.sub(r'(\d+(?:\.\d+)?)\s*k\b', lambda m: str(int(float(m.group(1)) * 1000)), t)
    return t

def extract_facts_from_text(text: str, current_facts: Dict[str, Any], pending_feature: str = None) -> Dict[str, Any]:
    """Rule-based & Regex NLU Extractor with On-The-Fly learning fallback for unseen vehicles."""
    facts = dict(current_facts)
    text_clean = normalize_user_text(text)
    
    # Extract any 4 to 7 digit numbers after normalization (e.g. 110000 from 110k)
    digits = re.findall(r'\b\d{4,7}\b', text_clean)
    parsed_nums = [float(d) for d in digits if int(d) != int(facts.get('pincode', 0))]
    
    # 0. Contextual Pending Feature Extraction
    if pending_feature == 'vehicle_price' and parsed_nums and 'vehicle_price' not in facts:
        facts['vehicle_price'] = parsed_nums[0]
    elif pending_feature == 'loan_amount' and parsed_nums and 'loan_amount' not in facts:
        facts['loan_amount'] = parsed_nums[0]
    elif pending_feature == 'net_salary' and parsed_nums and 'net_salary' not in facts:
        facts['net_salary'] = parsed_nums[0]
        
    # 1. Pincode Extraction (6 digits starting with 1-8, guarded by context to prevent 110000 price misidentification)
    pin_keywords = ['pincode', 'pin', 'zip', 'postal', 'area', 'location', 'sector']
    is_pin_context = pending_feature == 'pincode' or any(k in text_clean for k in pin_keywords)
    pin_match = re.search(r'\b([1-8]\d{5})\b', text_clean)
    if pin_match and 'pincode' not in facts and is_pin_context:
        facts['pincode'] = pin_match.group(1)
        
    # 2. General Loan Amount, Vehicle Price & Net Salary Extraction
    salary_keywords = ['salary', 'earn', 'income', 'pm', 'per month', 'monthly', 'make']
    is_salary_context = any(k in text_clean for k in salary_keywords)
    
    for num in parsed_nums:
        if pending_feature == 'vehicle_price' and 'vehicle_price' not in facts:
            facts['vehicle_price'] = num
        elif pending_feature == 'loan_amount' and 'loan_amount' not in facts:
            facts['loan_amount'] = num
        elif pending_feature == 'net_salary' and 'net_salary' not in facts:
            facts['net_salary'] = num
        elif 8000 <= num <= 500000:
            if is_salary_context or 'net_salary' not in facts:
                if 'net_salary' not in facts:
                    facts['net_salary'] = num
                elif 'loan_amount' not in facts and num != facts.get('net_salary'):
                    facts['loan_amount'] = num
        elif 15000 <= num <= 500000:
            if 'loan_amount' not in facts:
                facts['loan_amount'] = num
                
    # 3. Training Dataset Vehicles
    training_makes = {
        'JUPITER': ['jupiter', 'tvs jupiter'],
        'ACTIVA': ['activa', 'honda activa'],
        'SPLENDOR': ['splendor', 'hero splendor'],
        'APACHE': ['apache', 'tvs apache'],
        'PULSAR': ['pulsar', 'bajaj pulsar'],
        'BULLET': ['bullet', 'royal enfield', 'classic 350'],
        'ACCESS': ['access', 'suzuki access'],
        'SHINE': ['shine', 'honda shine'],
        'XL100': ['xl100', 'tvs xl']
    }
    
    if 'make_code' not in facts:
        for code, keywords in training_makes.items():
            if any(k in text_clean for k in keywords):
                facts['make_code'] = code
                break
                
        # If unseen vehicle mentioned, query Groq LLM on-the-fly and save to learned cache!
        if 'make_code' not in facts and len(text_clean) > 2 and not text_clean.isdigit():
            learned_info = lookup_unseen_vehicle_on_the_fly(text_clean)
            if learned_info:
                facts['make_code'] = learned_info['make_code']
                facts['vehicle_name'] = learned_info['canonical_name']
                facts['suggested_price'] = learned_info['baseline_price']
                
    if 'make_code' in facts and 'suggested_price' not in facts:
        facts['suggested_price'] = get_catalog_price_for_make(facts['make_code'], facts.get('vehicle_name'))
                
    # 4. Age Extraction
    age_match = re.search(r'\b(1[8-9]|[2-6]\d|70)\b', text_clean)
    if age_match and 'age' not in facts:
        facts['age'] = int(age_match.group(1))
            
    # 5. Employment Type Mapping (All 8 Dataset Categories)
    if 'employment_type' not in facts:
        if any(k in text_clean for k in ['freelanc', 'contractor', 'gig', 'npp', 'private work']):
            facts['employment_type'] = 'NPP'
        elif any(k in text_clean for k in ['farm', 'agri', 'kisan', 'crop']):
            facts['employment_type'] = 'AGR'
        elif any(k in text_clean for k in ['shop', 'trader', 'store', 'merchant', 'vendor']):
            facts['employment_type'] = 'NREGI'
        elif any(k in text_clean for k in ['student', 'college', 'study']):
            facts['employment_type'] = 'STU'
        elif any(k in text_clean for k in ['retire', 'pension']):
            facts['employment_type'] = 'PEN'
        elif any(k in text_clean for k in ['homemaker', 'housewife', 'non-earning']):
            facts['employment_type'] = 'NONEARNMEM'
        elif any(k in text_clean for k in ['doctor', 'lawyer', 'ca', 'consultant', 'professional']):
            facts['employment_type'] = 'SEP'
        elif any(k in text_clean for k in ['salaried', 'job', 'company', 'service', 'employee', 'work at', 'employed']):
            facts['employment_type'] = 'SAL'
        elif any(k in text_clean for k in ['self', 'business', 'own work']):
            facts['employment_type'] = 'SEP'
        
    return facts

# Standard Adverse Action Reason Codes & Human Explanations
REASON_CODES: Dict[str, tuple[str, Optional[str]]] = {
    "LTV": ("R02", "You're financing a large share of the vehicle's price."),
    "LTV_vs_Variant_Median": ("R02", "You're financing a large share of the vehicle's price."),
    "Down_Payment": ("R04", "The down payment is small relative to the vehicle's price."),
    "Down_Payment_Months": ("R04", "The down payment is small relative to the vehicle's price."),
    # R03 is resolved contextually -- see _age_reason(). A single fixed string
    # produced "your age falls outside the range we typically lend to" for
    # a 26-year-old, which is false: 26 is inside policy.
    "Age": ("R03", None),
    "Net_salary": ("R06", "Your stated monthly income is on the lower side for a loan this size."),
    "FOIR": ("R09", "The estimated monthly repayment would take up a large share of your income."),
    "Loan_Amount": ("R09", "The estimated monthly repayment would take up a large share of your income."),
    "PAST_LOANS_ACTIVE_NO_PAST_LOANS": ("R10", "We don't have any previous credit history with you to go on."),
    "PAST_LOANS_ACTIVE_PAST_LOANS_ACTIVE": ("R05", "You currently have another active loan."),
    "Prosperity_Score": ("R07", "The affordability profile of your area was a factor."),
    "Final_Tier_Num": ("R07", "The affordability profile of your area was a factor."),
    "Engine_CC": ("R08", "This is a higher-specification vehicle than we usually finance at this loan size."),
    "Vehicle_Value": ("R08", "This vehicle costs more than we usually finance at this income level."),
    "Price_vs_Variant_Median": ("R08", "This vehicle is priced above what's typical for this model."),
    "Engine_Vs_Prosperity": ("R07", "The affordability profile of your area was a factor."),
}
DEFAULT_REASON = ("R99", "A few details didn't line up with what we typically approve.")

def _age_reason(age: float) -> Optional[tuple[str, str]]:
    """Contextually evaluates age so normal adult ages (e.g., 26) are NOT misstated."""
    if age < 18 or age > 60:
        return ("R03", f"Applicant age ({int(age)}) falls outside our standard 18–60 lending eligibility bracket.")
    elif 18 <= age <= 21:
        return ("R03", "Younger applicants starting their credit profile may require additional credit verification.")
    else:
        # Age is inside standard policy (22-60) -> suppress age reason to avoid false adverse action!
        return None

def compute_decline_reasons(
    ltv: float,
    loan_amount: float,
    vehicle_price: float,
    net_salary: float,
    age: float,
    foir: float,
    loan_to_income: float,
    past_loans: str,
    prosperity_score: float,
    pincode_risk: float,
    engine_cc: float,
    down_payment_ratio: float,
    is_policy_guard: bool = False,
    collected_keys: set = None
) -> tuple[list[str], list[str]]:
    """
    Evaluates adverse action decline reasons using structured REASON_CODES.
    STRICT RULE: A reason code is ONLY included if its underlying data feature was explicitly collected from the user.
    Returns (decline_codes, decision_reasons).
    """
    if collected_keys is None:
        collected_keys = {'make_code', 'vehicle_price', 'loan_amount', 'net_salary', 'employment_type', 'resident_type', 'pincode', 'age'}

    reasons_map: Dict[str, str] = {}  # Map code -> reason string

    if is_policy_guard:
        if ltv > 100.0 and ('loan_amount' in collected_keys or 'vehicle_price' in collected_keys):
            code, msg = REASON_CODES["LTV"]
            reasons_map[code] = f"Requested loan amount exceeds vehicle on-road price (LTV {ltv:.1f}% > 100%)."
        if ('age' in collected_keys) and (age < 18 or age > 60):
            age_res = _age_reason(age)
            if age_res and age_res[1]:
                reasons_map[age_res[0]] = age_res[1]
    else:
        # Soft ML Model / Underwriting Policy Evaluation
        # Only evaluate features that were explicitly collected from the user!
        if ('loan_amount' in collected_keys or 'vehicle_price' in collected_keys) and ltv > 85.0:
            code, msg = REASON_CODES["LTV"]
            reasons_map[code] = msg
            
        if ('loan_amount' in collected_keys or 'vehicle_price' in collected_keys) and down_payment_ratio < 0.15:
            code, msg = REASON_CODES["Down_Payment"]
            reasons_map[code] = msg
            
        if ('net_salary' in collected_keys) and (loan_to_income > 0.35 or net_salary < 15000):
            code, msg = REASON_CODES["Net_salary"]
            reasons_map[code] = msg
            
        if ('net_salary' in collected_keys and 'loan_amount' in collected_keys) and foir > 0.40:
            code, msg = REASON_CODES["FOIR"]
            reasons_map[code] = msg
            
        if ('past_loans' in collected_keys) and past_loans == "PAST_LOANS_ACTIVE":
            code, msg = REASON_CODES["PAST_LOANS_ACTIVE_PAST_LOANS_ACTIVE"]
            reasons_map[code] = msg
            
        if ('pincode' in collected_keys) and (prosperity_score < 40.0 or pincode_risk > 0.06):
            code, msg = REASON_CODES["Prosperity_Score"]
            reasons_map[code] = msg
            
        if ('make_code' in collected_keys or 'vehicle_name' in collected_keys) and (engine_cc > 350.0 or vehicle_price > 300000):
            code, msg = REASON_CODES["Engine_CC"]
            reasons_map[code] = msg

        if 'age' in collected_keys:
            age_res = _age_reason(age)
            if age_res and age_res[1]:
                reasons_map[age_res[0]] = age_res[1]

    if not reasons_map:
        code, msg = DEFAULT_REASON
        reasons_map[code] = msg

    decline_codes = [f"{code}: {msg}" for code, msg in reasons_map.items()]
    decision_reasons = [f"[{code}] {msg}" for code, msg in reasons_map.items()]
    return decline_codes, decision_reasons

def generate_decline_explanation_with_groq(decline_codes: list[str], loan_amount: float, vehicle_price: float) -> str:
    """Uses Groq LLM to convert structured adverse action codes into a warm, human-readable natural language explanation."""
    if not groq_client:
        return "We regret that your loan application could not be approved at this time based on our underwriting policies. Please review the reason codes provided above or visit an ABC Credit branch."
    try:
        codes_text = "\n".join(f"  - {c}" for c in decline_codes)
        prompt = (
            f"You are ABC Credit's empathetic loan advisor AI. A loan application has been declined based on the following adverse action reason codes:\n"
            f"Loan Amount: ₹{loan_amount:,.0f}\nVehicle On-road Price: ₹{vehicle_price:,.0f}\n"
            f"Adverse Action Reasons:\n{codes_text}\n\n"
            f"Write a polite, empathetic, and clear 2-3 sentence explanation in simple English for the applicant.\n"
            f"Explain clearly what factor led to the decision and what constructive step they can take next.\n"
            f"Do NOT invent false facts or mention internal threshold numbers. Be warm, professional, and helpful."
        )
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Decline NL generation error: {e}")
        return "We regret that your application could not be approved at this time. Please review the reason codes above or contact an ABC Credit representative for assistance."

def evaluate_intermediate_stage(facts: Dict[str, Any]) -> tuple[float, str, bool]:
    """Scores candidate PD using available features.
    Returns (prob_default, decision, is_confident_early_exit)."""
    pincode_str = str(facts.get('pincode', '500090')).split('.')[0].zfill(6)
    make_code = str(facts.get('make_code', 'JUPITER')).upper()
    loan_amount = float(facts.get('loan_amount', 85000.0))
    net_salary = float(facts.get('net_salary', 45000.0))
    age = float(facts.get('age', 32.0))
    employment_type = str(facts.get('employment_type', 'SAL'))
    
    # Use actual vehicle_price from facts if available; otherwise estimate from engine_cc
    if 'vehicle_price' in facts and float(facts['vehicle_price']) > 0:
        vehicle_price = float(facts['vehicle_price'])
    else:
        vehicle_price = float(make_to_cc.get(make_code, 125.0)) * 900.0
        if vehicle_price < loan_amount:
            vehicle_price = loan_amount * 1.25
    ltv = min(max((loan_amount / vehicle_price) * 100.0, 10.0), 99.0)
    
    qualifications = 'GRAD'
    gender = 'Male'
    product_code = 'MC'
    resident_type = str(facts.get('resident_type', 'O'))
    past_loans = 'NO_PAST_LOANS'
    
    pincode_risk = pincode_risk_dict.get(str(int(pincode_str)) if pincode_str.isdigit() else "0", global_mean)
    make_risk = make_risk_dict.get(make_code, global_mean)
    prosperity_score = float(pin_to_score.get(pincode_str, overall_pros))
    engine_cc = float(make_to_cc.get(make_code, 125.0))
    
    asset_value = vehicle_price
    down_payment = max(asset_value - loan_amount, 0.0)
    down_payment_ratio = down_payment / (asset_value + 1e-5)
    loan_to_income = loan_amount / (net_salary * 12.0 + 1e-5)
    ltv_high_risk = 1 if ltv > 85.0 else 0
    estimated_emi = loan_amount * 0.045
    emi_to_income = estimated_emi / (net_salary + 1e-5)
    foir = (estimated_emi + (net_salary * 0.10)) / (net_salary + 1e-5)
    high_dti = 1 if loan_to_income > 0.35 else 0
    income_per_age = net_salary / (age + 1e-5)
    
    tier_num = 6 if pincode_risk > 0.06 else 5
    engine_vs_prosperity = engine_cc * prosperity_score
    
    row_data = {
        'Age': [age],
        'pincode_risk': [pincode_risk],
        'make_risk': [make_risk],
        'Loan_Amount': [loan_amount],
        'LTV': [ltv],
        'Net_salary': [net_salary],
        'Asset_Value': [asset_value],
        'Down_Payment': [down_payment],
        'Down_Payment_Ratio': [down_payment_ratio],
        'Loan_to_Income': [loan_to_income],
        'LTV_High_Risk': [ltv_high_risk],
        'Estimated_EMI': [estimated_emi],
        'EMI_to_Income': [emi_to_income],
        'FOIR': [foir],
        'High_DTI': [high_dti],
        'Income_Per_Age': [income_per_age],
        'Prosperity_Score': [prosperity_score],
        'Final_Tier_Num': [tier_num],
        'Engine_CC': [engine_cc],
        'Engine_Vs_Prosperity': [engine_vs_prosperity],
        'Qualifications': ['GRAD'],
        'Employment_Type': [employment_type],
        'Gender': ['Male'],
        'Product_Code': ['MC'],
        'Resident_Type': ['O'],
        'Final_Tier': ["06 Semi-Urban" if tier_num == 6 else "05 Urban"],
        'Make_Code': [make_code],
        'PAST_LOANS_ACTIVE': ['NO_PAST_LOANS']
    }
    
    X_input = pd.DataFrame(row_data)
    for c in cat_cols:
        X_input[c] = X_input[c].astype('category')
    X_input = X_input[feature_cols]
    
    prob_default = float(model.predict_proba(X_input)[0, 1])
    decision = "DECLINED" if prob_default >= decision_threshold else "APPROVED"
    
    # Early Exit Criteria: Only trigger early exit if model certainty is extremely high (PD <= 0.025 or PD >= 0.16).
    # Borderline profiles (0.025 < PD < 0.16) will continue collecting employment, residence, and pincode!
    num_facts = len([k for k in ['pincode', 'make_code', 'loan_amount', 'net_salary'] if k in facts])
    is_confident = (num_facts >= 3) and (prob_default <= 0.025 or prob_default >= 0.16)
    
    return prob_default, decision, is_confident

class LoanApplicationRequest(BaseModel):
    full_name: Optional[str] = "John Doe"
    phone_number: Optional[str] = "9876543210"
    email_address: Optional[str] = "applicant@example.com"
    
    gender: str = "Male"
    age: float = 32.0
    qualifications: str = "GRAD"
    employment_type: str = "SAL"
    net_salary: float = 45000.0
    resident_type: str = "O"
    pincode: Optional[str] = None
    past_loans_active: str = "NO_PAST_LOANS"
    
    product_code: str = "MC"
    make_code: str = "JUPITER"
    vehicle_name: Optional[str] = None
    vehicle_price: float = 110000.0
    loan_amount: float = 85000.0
    ltv: Optional[float] = None
    collected_fields: Optional[list[str]] = None

class EarlyExitCheckRequest(BaseModel):
    pincode: Optional[str] = "500090"
    net_salary: Optional[float] = 45000.0
    loan_amount: Optional[float] = 85000.0
    make_code: Optional[str] = "JUPITER"
    age: Optional[float] = 32.0
    employment_type: Optional[str] = "SAL"
    qualifications: Optional[str] = "GRAD"
    resident_type: Optional[str] = "O"

@app.post("/api/check-early-exit")
def check_early_exit(req: EarlyExitCheckRequest):
    facts = {}
    if req.pincode: facts['pincode'] = req.pincode
    if req.net_salary and req.net_salary > 0: facts['net_salary'] = req.net_salary
    if req.loan_amount and req.loan_amount > 0: facts['loan_amount'] = req.loan_amount
    if req.make_code: facts['make_code'] = req.make_code
    if req.age and req.age > 0: facts['age'] = req.age
    if req.employment_type: facts['employment_type'] = req.employment_type
    
    prob_default, decision, is_confident = evaluate_intermediate_stage(facts)
    return {
        'probability_of_default': round(prob_default, 4),
        'decision': decision,
        'is_confident_early_exit': is_confident
    }

VEHICLE_CATALOG_FILE = os.path.join(DATA_DIR, 'Vehicle_Catalog.csv')

def get_sanitized_vehicle_catalog():
    catalog = []
    if os.path.exists(VEHICLE_CATALOG_FILE):
        try:
            df_cat = pd.read_csv(VEHICLE_CATALOG_FILE)
            for _, r in df_cat.iterrows():
                catalog.append({
                    "model_description": str(r["Model_Description"]),
                    "model_variant": str(r["Model_Variant"]),
                    "make_code": str(r["Make_Code"]),
                    "typical_price": float(r["Typical_Price"]),
                    "engine_cc": float(r["Engine_CC"])
                })
        except Exception as e:
            print(f"Error loading vehicle catalog CSV: {e}")
    return catalog

@app.get("/api/vehicle-catalog")
def get_vehicle_catalog():
    return get_sanitized_vehicle_catalog()

@app.post("/api/chat")
def chat_turn(req: ChatTurnRequest):
    """Adaptive Conversational AI Chat Endpoint."""
    missing_order = ['make_code', 'vehicle_price', 'loan_amount', 'net_salary', 'employment_type', 'resident_type', 'pincode', 'age']
    next_missing = None
    for f in missing_order:
        if f not in req.collected_facts:
            next_missing = f
            break
            
    facts = extract_facts_with_groq(req.user_message, req.collected_facts, next_missing)
    prob_default, decision, is_confident = evaluate_intermediate_stage(facts)
    
    # Information Gain Cascade (Atomic Single-Question Order for Early Exit Evaluation)
    missing_order = ['make_code', 'vehicle_price', 'loan_amount', 'net_salary', 'employment_type', 'resident_type', 'pincode', 'age']
    next_missing = None
    
    for f in missing_order:
        if f not in facts:
            next_missing = f
            break
            
    if is_confident or next_missing is None:
        # Finalize Decision Card — pass ALL collected facts including vehicle_price and vehicle_name
        eval_req = LoanApplicationRequest(
            full_name=str(facts.get('full_name', 'Applicant')),
            pincode=str(facts['pincode']) if 'pincode' in facts else None,
            make_code=str(facts.get('make_code', 'JUPITER')),
            vehicle_name=str(facts.get('vehicle_name', facts.get('make_code', 'Vehicle'))),
            vehicle_price=float(facts.get('vehicle_price', 0.0)),
            loan_amount=float(facts.get('loan_amount', 85000.0)),
            net_salary=float(facts.get('net_salary', 45000.0)),
            age=float(facts.get('age', 32.0)),
            employment_type=str(facts.get('employment_type', 'SAL')),
            resident_type=str(facts.get('resident_type', 'O')),
            collected_fields=list(facts.keys())
        )
        res = evaluate_loan(eval_req)
        
        bot_message = (
            f"Thank you! Based on your financial profile, our automated credit engine has evaluated your application. "
            f"Final Decision: **{res['decision']}** (Approval Confidence Score: {res['approval_score']}/100)."
        )
        return {
            'status': 'complete',
            'bot_message': bot_message,
            'collected_facts': facts,
            'is_complete': True,
            'decision_result': res
        }
    else:
        # Prompt for Next Highest Information Gain Feature
        vehicle_display = str(facts.get('vehicle_name', facts.get('make_code', 'Vehicle'))).title()
        sug_price = float(facts.get('suggested_price', facts.get('vehicle_price', 110000.0)))
        prompts = {
            'make_code': "Which two-wheeler model are you planning to purchase? You can select from our dataset catalog dropdown or type any model in the chatbox below.",
            'vehicle_price': f"What is your actual negotiated on-road deal price (in ₹) for the **{vehicle_display}**?",
            'loan_amount': f"What is your requested loan amount (in ₹) for **{vehicle_display}**?",
            'net_salary': "What is your approximate net monthly salary / income (in ₹)?",
            'employment_type': "What is your employment sector?",
            'resident_type': "What is your residential status?",
            'pincode': "Could you share your 6-digit residential Pincode (or use GPS auto-detect)?",
            'age': "May I know your age?"
        }
        is_digression = facts.pop('_is_digression', False)
        validation_text = facts.pop('_validation_error', None)
        
        ask_text = prompts.get(next_missing, "Could you provide your net monthly income?")
        
        if validation_text:
            final_bot_msg = f"⚠️ {validation_text} {ask_text}"
        elif is_digression:
            final_bot_msg = f"I am your ABC Credit automated loan decision assistant. Let's focus on completing your instant loan pre-approval! {ask_text}"
        else:
            final_bot_msg = ask_text
        
        return {
            'status': 'in_progress',
            'bot_message': final_bot_msg,
            'collected_facts': facts,
            'is_complete': False,
            'next_feature': next_missing
        }

@app.post("/api/evaluate-loan")
def evaluate_loan(req: LoanApplicationRequest):
    start_time = time.time()
    
    # 1. Clean & Format Inputs
    age = float(req.age)
    net_salary = float(req.net_salary)
    loan_amount = float(req.loan_amount)
    vehicle_price = float(req.vehicle_price) if req.vehicle_price > 0 else loan_amount * 1.2
    
    if req.ltv is not None and req.ltv > 0:
        ltv = float(req.ltv)
    else:
        ltv = float((loan_amount / vehicle_price) * 100.0)
    # Do NOT clamp LTV — let policy guards below catch > 100
    ltv = max(ltv, 0.1)
    
    qualifications = req.qualifications.upper().strip()
    employment_type = req.employment_type.upper().strip()
    gender = req.gender.capitalize().strip()
    product_code = req.product_code.upper().strip()
    resident_type = req.resident_type.upper().strip()
    make_code = req.make_code.upper().strip()
    past_loans = req.past_loans_active.upper().strip()
    
    pincode_clean = str(req.pincode).split(".")[0].strip().zfill(6)
    pincode_val = float(pincode_clean) if pincode_clean.isdigit() else 0.0
    
    # ── HARD POLICY GUARDS (pre-model, instant reject) ──────────────────────
    if ltv > 100.0 or age < 18 or age > 60:
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        session_id = f"ABC-{int(time.time()*1000)}"
        masked_name = mask_name(req.full_name)
        masked_pin  = mask_pincode(pincode_clean)
        
        decline_codes, decision_reasons = compute_decline_reasons(
            ltv=ltv, loan_amount=loan_amount, vehicle_price=vehicle_price,
            net_salary=net_salary, age=age, foir=0.0, loan_to_income=0.0,
            past_loans=past_loans, prosperity_score=0.0, pincode_risk=0.0,
            engine_cc=0.0, down_payment_ratio=0.0, is_policy_guard=True
        )
        nl_explanation = generate_decline_explanation_with_groq(decline_codes, loan_amount, vehicle_price)
        hard_audit = {
            'session_id': session_id,
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            'applicant': {'masked_name': masked_name, 'age': age, 'masked_pincode': masked_pin},
            'financials': {'loan_amount': loan_amount, 'vehicle_price': vehicle_price, 'calculated_ltv': round(ltv, 2)},
            'ml_decisioning': {'decision': 'DECLINED', 'decline_codes': decline_codes, 'policy_guard': True, 'response_time_ms': elapsed_ms}
        }
        try:
            with open(AUDIT_LOG_FILE, 'a') as af:
                af.write(json.dumps(hard_audit) + "\n")
        except Exception:
            pass
        return {
            'status': 'success',
            'session_id': session_id,
            'decision': 'DECLINED',
            'approval_score': 0.0,
            'probability_of_default': 1.0,
            'optimal_threshold': decision_threshold,
            'response_time_ms': elapsed_ms,
            'policy_guard_triggered': True,
            'decline_codes': decline_codes,
            'natural_language_explanation': nl_explanation,
            'applicant_summary': {'name': masked_name, 'loan_amount': loan_amount, 'vehicle_price': vehicle_price, 'ltv': round(ltv, 1), 'estimated_emi': 0},
            'loan_details': {'requested_loan': loan_amount, 'vehicle_price': vehicle_price, 'vehicle_make': make_code, 'calculated_ltv': round(ltv, 1)},
            'risk_factors': {'LTV': f"{ltv:.1f}%", 'FOIR': '—', 'pincode_tier': '—'},
            'decision_reasons': decision_reasons,
            'next_steps': ['Revise requested loan amount below vehicle purchase price', 'Visit nearest ABC Credit branch for manual consultation'],
            'recommended_loan_limit': round(vehicle_price * 0.80 / 1000) * 1000
        }
    # ────────────────────────────────────────────────────────────────────────

    # 2. Risk Lookups
    pincode_clean = str(req.pincode).split(".")[0].strip().zfill(6) if (req.pincode and str(req.pincode).strip().lower() not in ['none', '', 'null']) else None
    
    if pincode_clean and pincode_clean.isdigit() and len(pincode_clean) == 6:
        pincode_val = float(pincode_clean)
        pincode_key = str(int(pincode_val))
        pincode_risk = pincode_risk_dict.get(pincode_key, global_mean)
        prosperity_score = float(pin_to_score.get(pincode_clean, overall_pros))
        masked_pin = mask_pincode(pincode_clean)
        tier_num = 6 if pincode_risk > 0.06 else 5
        tier_str = "Semi-Urban" if tier_num == 6 else "Urban"
    else:
        # Pincode was NOT provided by user — do not predict or assume a pincode
        pincode_risk = global_mean
        prosperity_score = overall_pros
        masked_pin = "Not Provided"
        tier_num = 5
        tier_str = "Not Provided"

    make_risk = make_risk_dict.get(make_code, global_mean)
    engine_cc = float(make_to_cc.get(make_code, 125.0))
    
    # 3. Derived Features
    asset_value = vehicle_price
    down_payment = max(asset_value - loan_amount, 0.0)
    down_payment_ratio = down_payment / (asset_value + 1e-5)
    loan_to_income = loan_amount / (net_salary * 12.0 + 1e-5)
    ltv_high_risk = 1 if ltv > 85.0 else 0
    estimated_emi = loan_amount * 0.045
    emi_to_income = estimated_emi / (net_salary + 1e-5)
    foir = (estimated_emi + (net_salary * 0.10)) / (net_salary + 1e-5)
    high_dti = 1 if loan_to_income > 0.35 else 0
    income_per_age = net_salary / (age + 1e-5)
    
    tier_num = 6 if pincode_risk > 0.06 else 5
    engine_vs_prosperity = engine_cc * prosperity_score
    
    row_data = {
        'Age': [age],
        'pincode_risk': [pincode_risk],
        'make_risk': [make_risk],
        'Loan_Amount': [loan_amount],
        'LTV': [ltv],
        'Net_salary': [net_salary],
        'Asset_Value': [asset_value],
        'Down_Payment': [down_payment],
        'Down_Payment_Ratio': [down_payment_ratio],
        'Loan_to_Income': [loan_to_income],
        'LTV_High_Risk': [ltv_high_risk],
        'Estimated_EMI': [estimated_emi],
        'EMI_to_Income': [emi_to_income],
        'FOIR': [foir],
        'High_DTI': [high_dti],
        'Income_Per_Age': [income_per_age],
        'Prosperity_Score': [prosperity_score],
        'Final_Tier_Num': [tier_num],
        'Engine_CC': [engine_cc],
        'Engine_Vs_Prosperity': [engine_vs_prosperity],
        'Qualifications': [qualifications],
        'Employment_Type': [employment_type],
        'Gender': [gender],
        'Product_Code': [product_code],
        'Resident_Type': [resident_type],
        'Final_Tier': ["06 Semi-Urban" if tier_num == 6 else "05 Urban"],
        'Make_Code': [make_code],
        'PAST_LOANS_ACTIVE': [past_loans]
    }
    
    X_input = pd.DataFrame(row_data)
    for c in cat_cols:
        X_input[c] = X_input[c].astype('category')
        
    X_input = X_input[feature_cols]
    
    # 4. Strict Binary Classification (APPROVED / DECLINED)
    prob_default = float(model.predict_proba(X_input)[0, 1])
    is_declined = prob_default >= decision_threshold
    decision = "DECLINED" if is_declined else "APPROVED"
    
    approval_score = float(max(0.0, min(100.0, (1.0 - prob_default) * 100.0)))
    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    
    # 5. Mask PII Data
    masked_name = mask_name(req.full_name)
    masked_phone = mask_phone(req.phone_number)
    masked_email = mask_email(req.email_address)
    masked_pin = mask_pincode(pincode_clean)
    
    # 6. Recommendation Engine
    max_recommended_loan = round(vehicle_price * 0.80 / 1000) * 1000 if is_declined else loan_amount
    suggested_down_payment = max(vehicle_price - max_recommended_loan, down_payment)
    estimated_emi_calc = round((loan_amount * 1.14 / 24) if not is_declined else (max_recommended_loan * 1.14 / 24))
    
    collected_keys = set(req.collected_fields) if req.collected_fields else {'make_code', 'vehicle_price', 'loan_amount', 'net_salary', 'employment_type', 'resident_type', 'pincode', 'age'}
    
    if is_declined:
        decline_codes, reasons = compute_decline_reasons(
            ltv=ltv, loan_amount=loan_amount, vehicle_price=vehicle_price,
            net_salary=net_salary, age=age, foir=foir, loan_to_income=loan_to_income,
            past_loans=past_loans, prosperity_score=prosperity_score, pincode_risk=pincode_risk,
            engine_cc=engine_cc, down_payment_ratio=down_payment_ratio, is_policy_guard=False,
            collected_keys=collected_keys
        )
        nl_explanation = generate_decline_explanation_with_groq(decline_codes, loan_amount, vehicle_price)
        next_steps = [
            f"Increase down payment to ₹{suggested_down_payment:,.0f} to reduce LTV below 85%",
            f"Apply for a revised loan amount of ₹{max_recommended_loan:,.0f}",
            "Add an earning co-applicant (Spouse/Parent) to increase household eligibility",
            "Visit nearest ABC Credit branch for manual loan officer consultation"
        ]
    else:
        decline_codes = []
        nl_explanation = None
        reasons = [
            "Excellent creditworthiness profile and income alignment",
            f"Healthy down payment of ₹{down_payment:,.0f} ({100-ltv:.1f}% equity margin)",
            f"High prosperity index ({prosperity_score:.2f}) and verified vehicle category"
        ]
        next_steps = [
            "Upload KYC (Aadhaar & PAN) to generate instant digital sanction letter",
            "Select flexible tenure (12 to 36 months) with dealership executive",
            "Schedule vehicle delivery at your nearest dealership"
        ]
        
    # 7. Audit Log Entry
    session_id = f"ABC-{int(time.time()*1000)}"
    audit_entry = {
        'session_id': session_id,
        'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'applicant': {
            'masked_name': masked_name,
            'masked_phone': masked_phone,
            'masked_email': masked_email,
            'masked_pincode': masked_pin,
            'gender': gender,
            'age': age,
            'qualifications': qualifications,
            'employment_type': employment_type,
            'resident_type': resident_type
        },
        'financials': {
            'net_salary': net_salary,
            'loan_amount': loan_amount,
            'vehicle_price': vehicle_price,
            'calculated_ltv': round(ltv, 2),
            'loan_to_income': round(loan_to_income, 3),
            'down_payment': round(down_payment, 2),
            'prosperity_score': round(prosperity_score, 4),
            'engine_cc': engine_cc
        },
        'vehicle': {
            'product_code': product_code,
            'make_code': make_code
        },
        'ml_decisioning': {
            'model_probability_default': round(prob_default, 4),
            'decision_threshold': decision_threshold,
            'decision': decision,
            'approval_score': round(approval_score, 1),
            'decline_codes': decline_codes,
            'response_time_ms': elapsed_ms
        }
    }
    
    try:
        with open(AUDIT_LOG_FILE, 'a') as f:
            f.write(json.dumps(audit_entry) + "\n")
    except Exception as e:
        print(f"Error writing audit log: {e}")
        
    return {
        'status': 'success',
        'session_id': session_id,
        'decision': decision,
        'approval_score': round(approval_score, 1),
        'probability_of_default': round(prob_default, 4),
        'optimal_threshold': decision_threshold,
        'response_time_ms': elapsed_ms,
        'applicant_summary': {
            'name': masked_name,
            'loan_amount': loan_amount,
            'vehicle_price': vehicle_price,
            'ltv': round(ltv, 1),
            'prosperity_score': round(prosperity_score, 2),
            'engine_cc': int(engine_cc),
            'estimated_emi': estimated_emi_calc
        },
        'loan_details': {
            'requested_loan': loan_amount,
            'vehicle_price': vehicle_price,
            'vehicle_make': req.vehicle_name if req.vehicle_name else make_code,
            'calculated_ltv': round(ltv, 1),
            'estimated_emi': estimated_emi_calc
        },
        'applicant': {
            'masked_name': masked_name,
            'masked_phone': masked_phone,
            'masked_email': masked_email,
            'masked_pincode': masked_pin
        },
        'risk_factors': {
            'LTV': f"{ltv:.1f}%",
            'FOIR': f"{foir*100:.1f}%",
            'pincode_tier': tier_str,
            'probability_default': f"{prob_default*100:.2f}%",
            'optimal_threshold': f"{decision_threshold*100:.2f}%"
        },
        'decision_reasons': reasons,
        'decline_codes': decline_codes,
        'natural_language_explanation': nl_explanation,
        'next_steps': next_steps,
        'recommended_loan_limit': max_recommended_loan
    }

@app.get("/api/health")
def health_check():
    return {
        'status': 'healthy',
        'model_loaded': True,
        'model_metrics': {
            'roc_auc': metadata['roc_auc'],
            'accuracy': metadata['accuracy'],
            'approved_recall': metadata.get('approved_recall', 0.978),
            'decision_threshold': metadata['decision_threshold']
        },
        'sla_targets': {
            'completion_time_limit': '1.5 minutes',
            'decisioning_time_sla': '< 0.5 seconds'
        }
    }

@app.get("/api/audit-logs")
def get_audit_logs(limit: int = 20):
    logs = []
    if os.path.exists(AUDIT_LOG_FILE):
        with open(AUDIT_LOG_FILE, 'r') as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                try:
                    logs.append(json.loads(line.strip()))
                except Exception:
                    pass
    return {'total_logs': len(logs), 'logs': logs[::-1]}

app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")

@app.get("/")
def read_root():
    index_file = os.path.join(PUBLIC_DIR, 'index.html')
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "ABC Credit Chatbot API Server is Running"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
