"""
AI Healthcare Assistant (Streamlit)
-----------------------------------
Run:  streamlit run healthcare_assistant.py

Features
  * Basic info (name, age, gender) + current symptoms
  * Interactive full-body diagram (front / back) with anatomical regions
  * Summary of potential causes, tests to confirm them, treatments & wellness tips
      - Works fully offline with a built-in knowledge base
      - Optional: add an Anthropic API key for a richer, personalised AI report
  * Emergency detection -> asks for city/town -> lists nearby hospitals
    (OpenStreetMap Nominatim + Overpass, no API key needed)

DISCLAIMER: Educational tool only. Not a medical diagnosis.
"""

import math
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Healthcare Assistant", page_icon="🩺", layout="wide")

AI_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
UA = {"User-Agent": "HealthcareAssistantDemo/1.0 (streamlit app)"}

# ---------------------------------------------------------------------------
# 1. INTERACTIVE BODY DIAGRAM & REGION DEFINITIONS
# ---------------------------------------------------------------------------

def sym(base, x, y, size, view):
    """Return a Right/Left pair of regions mirrored around the body's centre line."""
    right_x = -x if view == "Front" else x
    return [
        (f"Right {base}", right_x, y, size),
        (f"Left {base}", -right_x, y, size),
    ]


def regions_for(view):
    """Define interactive clickable hotspots for anatomical regions."""
    r = []
    if view == "Front":
        r += [("Head", 0, 9.9, 32), ("Neck", 0, 9.15, 20)]
        r += sym("Shoulder", 1.15, 8.65, 22, view)
        r += [("Chest", 0, 8.1, 38)]
        r += sym("Upper Arm", 1.25, 7.5, 22, view)
        r += [("Upper Abdomen", 0, 7.0, 34), ("Lower Abdomen", 0, 6.0, 34)]
        r += sym("Forearm", 1.42, 5.8, 20, view)
        r += sym("Hand", 1.5, 4.5, 20, view)
        r += [("Pelvis / Groin", 0, 5.15, 28)]
        r += sym("Thigh", 0.42, 3.9, 28, view)
        r += sym("Knee", 0.42, 2.85, 22, view)
        r += sym("Shin", 0.42, 1.65, 24, view)
        r += sym("Foot", 0.45, 0.3, 20, view)
    else:
        r += [("Back of Head", 0, 9.9, 32), ("Back of Neck", 0, 9.15, 20)]
        r += sym("Shoulder Blade", 0.6, 8.4, 24, view)
        r += [("Upper Back", 0, 8.3, 24), ("Mid Back", 0, 7.1, 36), ("Lower Back", 0, 5.9, 36)]
        r += sym("Back of Arm", 1.25, 7.0, 22, view)
        r += sym("Forearm", 1.42, 5.7, 20, view)
        r += sym("Hand", 1.5, 4.5, 20, view)
        r += [("Buttocks", 0, 4.95, 34)]
        r += sym("Hamstring", 0.42, 3.75, 28, view)
        r += sym("Knee", 0.42, 2.8, 20, view)
        r += sym("Calf", 0.42, 1.65, 26, view)
        r += sym("Heel", 0.42, 0.3, 20, view)
    return r


def body_figure(view, selected):
    """Generates an interactive Plotly vector figure with accurate body outline and custom markers."""
    fig = go.Figure()
    
    # Visual Styling
    body_fill = "rgba(226, 232, 240, 0.65)"
    body_line = "rgba(71, 85, 105, 0.85)"
    grid_line = "rgba(148, 163, 184, 0.3)"

    def path_shape(path_svg):
        fig.add_shape(type="path", path=path_svg, line=dict(color=body_line, width=1.8),
                      fillcolor=body_fill, layer="below")

    def rect(x0, y0, x1, y1):
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color=body_line, width=1.5), fillcolor=body_fill, layer="below")

    # Anatomical silhouette outlines
    # Head & Neck
    fig.add_shape(type="circle", x0=-0.52, y0=9.3, x1=0.52, y1=10.45,
                  line=dict(color=body_line, width=1.8), fillcolor=body_fill, layer="below")
    rect(-0.18, 8.95, 0.18, 9.35)

    # Torso & Hips
    path_shape("M -0.95 8.95 L 0.95 8.95 L 0.82 4.9 L 0.88 4.7 L -0.88 4.7 L -0.82 4.9 Z")

    # Arms and Legs
    for s in (-1, 1):
        rect(min(s * 1.0, s * 1.4), 6.85, max(s * 1.0, s * 1.4), 8.85)     # Upper arm
        rect(min(s * 1.2, s * 1.58), 4.95, max(s * 1.2, s * 1.58), 6.75)   # Forearm
        fig.add_shape(type="circle", x0=s * 1.48 - 0.22, y0=4.15, x1=s * 1.48 + 0.22, y1=4.88,
                      line=dict(color=body_line, width=1.5), fillcolor=body_fill, layer="below") # Hand
        rect(min(s * 0.08, s * 0.74), 2.9, max(s * 0.08, s * 0.74), 4.7)   # Thigh
        rect(min(s * 0.16, s * 0.66), 0.65, max(s * 0.16, s * 0.66), 2.8)  # Shin / Calf
        rect(min(s * 0.12, s * 0.72), 0.05, max(s * 0.12, s * 0.72), 0.55) # Foot

    regs = regions_for(view)
    
    # Styling node states
    colors = []
    line_colors = []
    line_widths = []
    hover_texts = []
    
    for n, *_ in regs:
        is_sel = n in selected
        colors.append("rgba(239, 68, 68, 0.9)" if is_sel else "rgba(14, 165, 233, 0.45)")
        line_colors.append("#B91C1C" if is_sel else "#0284C7")
        line_widths.append(2.5 if is_sel else 1.2)
        hover_texts.append(f"<b>{'🔴 ' if is_sel else '🔵 '}{n}</b><br>Click to toggle selection")

    fig.add_trace(go.Scatter(
        x=[r[1] for r in regs],
        y=[r[2] for r in regs],
        mode="markers+text",
        text=[r[0] if r[0] in selected else "" for r in regs],
        textposition="top center",
        textfont=dict(size=11, color="#1E293B", family="sans-serif"),
        customdata=[r[0] for r in regs],
        hovertext=hover_texts,
        hovertemplate="%{hovertext}<extra></extra>",
        marker=dict(
            size=[r[3] for r in regs],
            color=colors,
            line=dict(color=line_colors, width=line_widths),
            opacity=0.92
        ),
        selected=dict(marker=dict(opacity=1)),
        unselected=dict(marker=dict(opacity=0.92)),
    ))

    # Title & View annotation
    fig.add_annotation(
        x=0, y=10.95,
        text=f"<b>Interactive Map — {view} View</b>",
        showarrow=False,
        font=dict(size=15, color="#0F172A")
    )

    fig.update_xaxes(visible=False, range=[-2.2, 2.2], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-0.2, 11.3], fixedrange=True, scaleanchor="x")
    fig.update_layout(
        height=630,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        dragmode="select",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        clickmode="event+select"
    )
    return fig


def show_chart(fig, key):
    """Render interactive chart and capture select events with backwards compatibility."""
    import inspect
    params = inspect.signature(st.plotly_chart).parameters
    if "on_select" not in params:
        return None
    kw = dict(key=key, on_select="rerun", selection_mode="points", config={"displayModeBar": False})
    if "width" in params:
        kw["width"] = "stretch"
    else:
        kw["use_container_width"] = True
    return st.plotly_chart(fig, **kw)


def clicked_names(event, view):
    """Extract selected region names from Plotly interaction events."""
    try:
        pts = event.selection.points
    except Exception:
        try:
            pts = event["selection"]["points"]
        except Exception:
            return []
    names_in_view = [r[0] for r in regions_for(view)]
    out = []
    for pt in pts or []:
        n = pt.get("customdata") or pt.get("text")
        if isinstance(n, (list, tuple)):
            n = n[0] if n else None
        if not n:
            idx = pt.get("point_index", pt.get("point_number"))
            if isinstance(idx, int) and 0 <= idx < len(names_in_view):
                n = names_in_view[idx]
        if n:
            out.append(n)
    return out


ALL_REGIONS = list(dict.fromkeys([r[0] for v in ("Front", "Back") for r in regions_for(v)]))

# ---------------------------------------------------------------------------
# 2. KNOWLEDGE BASE  (name, likelihood, why, tests, self-care)
# likelihood: C = common, L = less common, S = serious / rule out
# ---------------------------------------------------------------------------
KB = {
    "head": [
        ("Tension-type headache", "C", "Stress, poor posture, dehydration, eye strain or lack of sleep.",
         ["Clinical exam", "Blood pressure check", "Eye test"],
         ["Hydrate and rest", "Short breaks from screens", "Neck/shoulder stretches", "OTC pain relief as per label"]),
        ("Migraine", "C", "Throbbing, often one-sided pain with nausea or light/sound sensitivity.",
         ["Neurological exam", "Headache diary", "MRI only if symptoms are atypical"],
         ["Rest in a dark, quiet room", "Keep regular sleep & meals", "Track triggers (food, stress, hormones)"]),
        ("Sinusitis", "C", "Facial pressure, blocked nose, pain worse when bending forward.",
         ["ENT exam", "Nasal endoscopy", "CT sinuses if recurrent"],
         ["Saline nasal rinse", "Steam inhalation", "Plenty of fluids"]),
        ("Serious intracranial cause (bleed, meningitis, stroke)", "S",
         "Rare, but suspect if headache is sudden/severe ('worst ever'), or with fever + stiff neck, confusion or weakness.",
         ["CT/MRI brain", "Lumbar puncture", "Blood tests"], ["Needs urgent medical evaluation"]),
    ],
    "neck": [
        ("Muscle strain / poor posture", "C", "Long hours at a desk or phone, awkward sleeping position.",
         ["Clinical exam"], ["Gentle stretching", "Warm compress", "Ergonomic workstation", "Supportive pillow"]),
        ("Cervical spondylosis / pinched nerve", "C", "Age-related disc/joint wear; may cause arm tingling.",
         ["Cervical spine X-ray", "MRI", "Nerve conduction study"], ["Physiotherapy", "Posture correction", "Avoid heavy lifting"]),
        ("Throat / lymph node infection", "L", "Swollen tender glands, sore throat, fever.",
         ["Throat swab", "CBC", "Neck ultrasound if swelling persists"], ["Warm fluids", "Rest", "Gargle with warm salt water"]),
    ],
    "chest": [
        ("Muscle strain / costochondritis", "C", "Pain that changes with movement, breathing or pressing on the chest wall.",
         ["Clinical exam", "ECG to exclude heart cause"], ["Rest from strenuous activity", "Warm compress", "OTC pain relief as per label"]),
        ("Acid reflux (GERD)", "C", "Burning behind the breastbone, worse after meals or lying down.",
         ["Clinical assessment", "Endoscopy if persistent", "H. pylori test"],
         ["Smaller meals", "Avoid spicy/fatty food & late dinners", "Raise head of bed", "Antacids as per label"]),
        ("Anxiety / panic episode", "L", "Chest tightness with palpitations, fast breathing, sense of dread.",
         ["Clinical evaluation", "ECG", "Thyroid test (TSH)"], ["Slow diaphragmatic breathing", "Regular exercise", "Limit caffeine", "Consider counselling"]),
        ("Lung infection (pneumonia / pleurisy)", "L", "Cough, fever, pain on deep breaths.",
         ["Chest X-ray", "CBC & CRP", "Sputum test", "Pulse oximetry"], ["Rest & fluids", "See a doctor for possible antibiotics"]),
        ("Heart-related pain (angina / heart attack)", "S",
         "Pressure, squeezing or heaviness, possibly spreading to arm/jaw, with sweating or breathlessness.",
         ["ECG", "Troponin blood test", "Echocardiogram", "Stress test / angiography"],
         ["Treat as an emergency if pain is severe, sudden or accompanied by breathlessness"]),
    ],
    "abdomen_upper": [
        ("Gastritis / peptic ulcer", "C", "Burning or gnawing pain, linked to meals, NSAIDs, alcohol or H. pylori.",
         ["H. pylori breath/stool test", "Upper GI endoscopy", "CBC"],
         ["Avoid spicy food, alcohol, smoking", "Small frequent meals", "Avoid NSAIDs unless advised"]),
        ("Indigestion / reflux", "C", "Bloating, fullness, burping after heavy meals.",
         ["Clinical assessment", "Ultrasound abdomen if persistent"], ["Eat slowly", "Light dinner", "Walk after meals"]),
        ("Gallstones / gallbladder inflammation", "L", "Pain under the right ribs after fatty meals, may spread to shoulder.",
         ["Abdominal ultrasound", "Liver function tests", "CBC"], ["Low-fat diet", "Medical review (may need surgery)"]),
        ("Pancreatitis", "S", "Severe upper abdominal pain radiating to the back, nausea/vomiting.",
         ["Serum lipase/amylase", "CT abdomen"], ["Needs urgent medical care"]),
    ],
    "abdomen_lower": [
        ("Gastroenteritis (stomach infection)", "C", "Cramps with diarrhoea, nausea, possibly fever.",
         ["Stool test", "Electrolytes if dehydrated"], ["ORS / plenty of fluids", "Bland diet (rice, banana, toast)", "Hand hygiene"]),
        ("Irritable bowel syndrome / constipation", "C", "Cramping, bloating, altered bowel habits, stress-linked.",
         ["Clinical history", "Stool tests", "Coeliac screen"], ["High-fibre diet", "Hydration", "Regular exercise", "Stress management"]),
        ("Urinary tract infection", "C", "Burning urination, frequency, lower abdominal discomfort.",
         ["Urinalysis", "Urine culture"], ["Drink water", "See a doctor for antibiotics if confirmed"]),
        ("Appendicitis", "S", "Pain starting near the navel, moving to lower RIGHT abdomen, with fever/vomiting.",
         ["CBC & CRP", "Abdominal ultrasound / CT"], ["Needs urgent surgical assessment"]),
    ],
    "pelvis": [
        ("Urinary tract infection", "C", "Burning urination, urgency, pelvic pressure.", ["Urinalysis", "Urine culture"],
         ["Hydrate well", "Medical review for antibiotics"]),
        ("Inguinal hernia / groin strain", "L", "Bulge or aching in the groin, worse on lifting or coughing.",
         ["Physical exam", "Groin ultrasound"], ["Avoid heavy lifting", "Surgical consult if bulge persists"]),
        ("Kidney / ureteric stone", "L", "Colicky pain radiating from flank to groin, blood in urine.",
         ["Urinalysis", "Ultrasound KUB / CT KUB"], ["Fluids", "Medical review for pain control"]),
    ],
    "back_upper": [
        ("Muscle strain / posture", "C", "Hunching over desks/phones, heavy bags, overuse.",
         ["Clinical exam"], ["Posture breaks every 30–45 min", "Heat therapy", "Shoulder-blade stretches"]),
        ("Thoracic spine joint dysfunction", "L", "Stiffness and localized pain, worse with twisting.",
         ["Spine X-ray", "MRI if nerve symptoms"], ["Physiotherapy", "Mobility exercises"]),
        ("Referred pain (heart, gallbladder, lungs)", "S", "Back pain that comes with chest pain, breathlessness or sweating.",
         ["ECG", "Chest X-ray", "Abdominal ultrasound"], ["Seek urgent care if with chest symptoms"]),
    ],
    "back_lower": [
        ("Mechanical low back strain", "C", "Lifting, prolonged sitting, weak core muscles.",
         ["Clinical exam", "X-ray only if persistent/injury"], ["Stay gently active (avoid bed rest)", "Heat packs", "Core-strengthening", "Ergonomic seating"]),
        ("Slipped disc / sciatica", "C", "Pain radiating down the leg, tingling or numbness.",
         ["Neurological exam", "MRI lumbar spine", "Nerve conduction study"], ["Physiotherapy", "Avoid heavy lifting", "Medical review"]),
        ("Kidney stone / kidney infection", "L", "Flank pain, fever, painful urination, blood in urine.",
         ["Urinalysis & culture", "Ultrasound / CT KUB", "Blood creatinine"], ["Drink fluids", "Needs medical evaluation"]),
        ("Cauda equina / spinal cord compression", "S",
         "Back pain with leg weakness, numbness in groin/saddle area, or loss of bladder/bowel control.",
         ["Urgent MRI spine"], ["Emergency – go to hospital immediately"]),
    ],
    "shoulder": [
        ("Rotator cuff strain / tendinitis", "C", "Pain lifting the arm or reaching overhead; overuse or sleeping on the shoulder.",
         ["Clinical exam", "Shoulder ultrasound", "MRI"], ["Relative rest", "Ice then heat", "Physio strengthening"]),
        ("Frozen shoulder", "L", "Gradual stiffness and loss of motion, often with diabetes.",
         ["Clinical exam", "X-ray", "HbA1c / blood sugar"], ["Gentle range-of-motion exercises", "Physiotherapy"]),
        ("Referred pain (neck nerve, heart – esp. left)", "S", "Left shoulder pain with chest pressure or breathlessness.",
         ["Cervical spine X-ray", "ECG"], ["Treat as emergency if with chest symptoms"]),
    ],
    "arm": [
        ("Muscle strain / tendinitis (e.g., tennis elbow)", "C", "Repetitive use or sudden overload.",
         ["Clinical exam", "Ultrasound"], ["Rest", "Ice for 15 min", "Gradual stretching & strengthening"]),
        ("Nerve compression (cubital / carpal tunnel)", "L", "Tingling or numbness in fingers, worse at night.",
         ["Nerve conduction study", "Vitamin B12 level"], ["Wrist/elbow splint at night", "Ergonomic changes"]),
        ("Fracture / sprain", "L", "Pain after a fall or injury with swelling or deformity.",
         ["X-ray"], ["Immobilise and get examined"]),
    ],
    "hand": [
        ("Carpal tunnel syndrome", "C", "Numb/tingling thumb, index and middle fingers, nocturnal symptoms.",
         ["Nerve conduction study", "Thyroid test (TSH)", "HbA1c"], ["Wrist splint", "Regular breaks from typing", "Nerve glide exercises"]),
        ("Arthritis (osteoarthritis / rheumatoid)", "C", "Joint stiffness, swelling, worse in the morning.",
         ["ESR/CRP", "RA factor & anti-CCP", "Hand X-ray"], ["Warm water soaks", "Joint-friendly exercise", "Rheumatology review"]),
        ("Tendon strain / trigger finger", "L", "Clicking, locking or pain when bending fingers.", ["Clinical exam", "Ultrasound"],
         ["Rest the hand", "Gentle stretching"]),
    ],
    "hip": [
        ("Piriformis syndrome / sciatica", "C", "Deep buttock pain, may travel down the leg.",
         ["Clinical exam", "MRI lumbar spine / pelvis"], ["Piriformis stretches", "Avoid prolonged sitting", "Physiotherapy"]),
        ("Hip bursitis / osteoarthritis", "C", "Pain on the outer hip, worse lying on that side or climbing stairs.",
         ["Hip X-ray", "Ultrasound", "MRI"], ["Weight management", "Low-impact exercise (swimming, cycling)"]),
    ],
    "thigh": [
        ("Muscle strain", "C", "Sudden pain during sport or overexertion.", ["Clinical exam", "Ultrasound"],
         ["RICE (rest, ice, compression, elevation)", "Gradual return to activity"]),
        ("Referred pain from spine (sciatica)", "L", "Pain/tingling from lower back down the leg.", ["MRI lumbar spine"],
         ["Physiotherapy", "Medical review"]),
        ("Deep vein thrombosis (DVT)", "S", "One-sided swelling, warmth, redness – especially after long immobility or travel.",
         ["D-dimer", "Doppler ultrasound of leg veins"], ["Seek urgent care if swelling is one-sided"]),
    ],
    "knee": [
        ("Osteoarthritis", "C", "Gradual pain and stiffness, worse on stairs or after rest.",
         ["Knee X-ray", "Clinical exam"], ["Weight management", "Quadriceps strengthening", "Low-impact exercise", "Hot/cold packs"]),
        ("Ligament / meniscus injury", "C", "Twisting injury, swelling, locking or giving way.",
         ["MRI knee", "Clinical stability tests"], ["RICE", "Avoid twisting activity", "Orthopaedic review"]),
        ("Patellofemoral pain / bursitis", "C", "Pain around the kneecap, worse climbing stairs or squatting.",
         ["Clinical exam", "Ultrasound"], ["Quadriceps and hip exercises", "Supportive footwear"]),
        ("Gout / septic arthritis", "S", "Hot, red, very swollen joint, possibly with fever.",
         ["Uric acid", "Joint fluid analysis", "CBC & CRP"], ["Needs prompt medical assessment"]),
    ],
    "lower_leg": [
        ("Shin splints / calf strain", "C", "Pain after running or new exercise.", ["Clinical exam", "X-ray if suspect stress fracture"],
         ["Rest", "Ice", "Supportive footwear", "Gradual training increase"]),
        ("Cramps (dehydration / electrolyte imbalance)", "C", "Sudden tightening, often at night.",
         ["Electrolytes (Na, K, Mg, Ca)", "Vitamin D"], ["Hydration", "Stretch calves before bed", "Balanced diet"]),
        ("Varicose veins / venous insufficiency", "L", "Heaviness, visible veins, swelling by evening.",
         ["Venous Doppler ultrasound"], ["Elevate legs", "Compression stockings", "Regular walking"]),
        ("Deep vein thrombosis (DVT)", "S", "Calf pain with one-sided swelling and warmth.",
         ["D-dimer", "Doppler ultrasound"], ["Seek urgent care"]),
    ],
    "foot": [
        ("Plantar fasciitis", "C", "Heel pain on first steps in the morning.", ["Clinical exam", "Foot X-ray / ultrasound"],
         ["Calf & plantar stretches", "Cushioned footwear", "Ice roll under arch"]),
        ("Sprain / stress fracture", "C", "Pain after twisting or repetitive impact.", ["X-ray", "MRI if X-ray normal"],
         ["RICE", "Avoid weight-bearing if painful", "Medical review"]),
        ("Gout", "L", "Sudden intense pain, redness at the big toe.", ["Serum uric acid", "Joint fluid analysis"],
         ["Fluids", "Limit red meat, alcohol, sugary drinks"]),
        ("Diabetic / peripheral neuropathy", "L", "Burning, numbness or tingling in the feet.",
         ["HbA1c / fasting glucose", "Vitamin B12", "Nerve conduction study"], ["Daily foot checks", "Blood-sugar control"]),
    ],
}

REGION_GROUPS = [
    ("back of head", "head"), ("neck", "neck"), ("shoulder blade", "back_upper"), ("shoulder", "shoulder"),
    ("lower back", "back_lower"), ("mid back", "back_upper"), ("upper back", "back_upper"),
    ("buttock", "hip"), ("chest", "chest"), ("upper abdomen", "abdomen_upper"),
    ("lower abdomen", "abdomen_lower"), ("pelvis", "pelvis"), ("forearm", "arm"), ("upper arm", "arm"),
    ("back of arm", "arm"), ("hand", "hand"), ("thigh", "thigh"), ("hamstring", "thigh"),
    ("knee", "knee"), ("shin", "lower_leg"), ("calf", "lower_leg"), ("foot", "foot"),
    ("heel", "foot"), ("head", "head"),
]

# Associated-symptom modifiers
SYMPTOM_KB = {
    "Fever": ("Infection (viral / bacterial)", "C", "Fever suggests the body is fighting an infection.",
              ["CBC", "CRP / ESR", "Blood/urine cultures if persistent", "Malaria/dengue tests if in endemic area & high fever"],
              ["Rest", "Fluids", "Paracetamol as per label", "See a doctor if fever > 3 days"]),
    "Fatigue": ("Anaemia / thyroid / vitamin deficiency / diabetes", "C", "Persistent tiredness has many treatable causes.",
                ["CBC & ferritin", "TSH", "HbA1c / fasting glucose", "Vitamin B12 & D"],
                ["Sleep 7–9 h", "Iron/protein-rich diet", "Gentle daily exercise"]),
    "Dizziness": ("Low BP / dehydration / inner-ear problem", "C", "Light-headedness or spinning sensation.",
                  ["BP (sitting & standing)", "Blood sugar", "CBC", "ECG", "Hearing/vestibular tests"],
                  ["Hydrate", "Rise slowly from lying/sitting", "Avoid driving until it settles"]),
    "Nausea / vomiting": ("Gastroenteritis / food intolerance / migraine", "C", "Common with stomach infections or migraines.",
                          ["Stool test", "Electrolytes", "Pregnancy test (if applicable)"],
                          ["Small sips of ORS", "Bland foods", "Avoid greasy meals"]),
    "Cough": ("Viral respiratory infection / asthma / allergy", "C", "Persistent cough may indicate airway inflammation.",
              ["Chest X-ray", "Spirometry", "Pulse oximetry"], ["Warm fluids & honey (adults)", "Steam", "Avoid smoke & dust"]),
    "Rash": ("Allergic reaction / dermatitis / viral rash", "C", "Skin changes with other symptoms.",
             ["Skin exam", "Allergy tests", "CBC"], ["Avoid known triggers", "Gentle moisturiser", "Do not scratch"]),
    "Swelling": ("Inflammation / fluid retention", "L", "Swelling may signal injury, infection, or heart/kidney/liver issues.",
                 ["Ultrasound", "Kidney & liver function tests", "Urine protein"], ["Elevate the area", "Reduce salt", "Medical review"]),
    "Numbness / tingling": ("Nerve compression / B12 deficiency / diabetes", "L", "Altered sensation suggests nerve involvement.",
                            ["Vitamin B12", "HbA1c", "Nerve conduction study"], ["Avoid prolonged pressure on the area", "Medical review"]),
}

RED_FLAG_SYMPTOMS = {
    "Shortness of breath", "Chest pain / pressure", "Sudden weakness / numbness on one side",
    "Confusion / slurred speech", "Fainting / loss of consciousness", "Severe bleeding",
    "Coughing up blood", "Sudden severe headache",
}
RED_FLAG_WORDS = [
    "can't breathe", "cannot breathe", "difficulty breathing", "unconscious", "passed out", "seizure",
    "stroke", "heart attack", "crushing", "severe bleeding", "vomiting blood", "blood in vomit",
    "suicid", "self-harm", "kill myself", "overdose", "poison", "paralys", "slurred",
    "worst headache", "anaphyla", "swollen tongue", "throat closing",
]
LIKELIHOOD = {"C": "🟢 Common", "L": "🟡 Less common", "S": "🔴 Serious – must be ruled out"}


# ---------------------------------------------------------------------------
# 3. ANALYSIS
# ---------------------------------------------------------------------------
def group_of(region):
    r = region.lower()
    for key, grp in REGION_GROUPS:
        if key in r:
            return grp
    return None


def detect_emergency(p):
    reasons = []
    if p["self_flag"]:
        reasons.append("You indicated this may be an emergency.")
    hit = sorted(set(p["assoc"]) & RED_FLAG_SYMPTOMS)
    if hit:
        reasons.append("Warning-sign symptom(s): " + ", ".join(hit).lower() + ".")
    txt = p["description"].lower()
    words = [w for w in RED_FLAG_WORDS if w in txt]
    if words:
        reasons.append("Your description mentions potentially urgent wording.")
    if p["severity"] >= 9:
        reasons.append(f"Pain/discomfort rated very high ({p['severity']}/10).")
    groups = {group_of(r) for r in p["regions"]}
    if "chest" in groups and (p["age"] >= 40 or p["severity"] >= 7 or "Sweating" in p["assoc"]):
        if {"Shortness of breath", "Chest pain / pressure", "Sweating"} & set(p["assoc"]) or p["severity"] >= 7:
            reasons.append("Chest discomfort together with other risk factors can signal a heart problem.")
    if "Sweating" in p["assoc"] and ("chest" in groups or "shoulder" in groups):
        reasons.append("Sweating with chest/shoulder discomfort can be a heart-attack warning sign.")
    if "back_lower" in groups and "Loss of bladder/bowel control" in p["assoc"]:
        reasons.append("Back pain with loss of bladder/bowel control needs immediate care.")
    return list(dict.fromkeys(reasons))


def rule_based_causes(p):
    causes, seen = [], set()

    def add(item, source):
        name = item[0]
        if name in seen:
            return
        seen.add(name)
        causes.append(dict(name=name, like=item[1], why=item[2], tests=item[3], care=item[4], source=source))

    groups = []
    for reg in p["regions"]:
        g = group_of(reg)
        if g and g not in [x[0] for x in groups]:
            groups.append((g, reg))
        elif g:
            idx = [x[0] for x in groups].index(g)
            if reg not in groups[idx][1]:
                groups[idx] = (g, groups[idx][1] + ", " + reg)
    for g, regs in groups:
        for item in KB.get(g, []):
            add(item, regs)

    # Gender / age specific additions
    gset = {g for g, _ in groups}
    if p["gender"] == "Female" and gset & {"abdomen_lower", "pelvis"}:
        add(("Menstrual cramps / ovarian cyst / endometriosis", "C",
             "Cyclical lower-abdominal or pelvic pain in people who menstruate.",
             ["Pelvic ultrasound", "Urine pregnancy test (hCG)", "CA-125 / hormonal tests if advised"],
             ["Heat pack", "Gentle exercise", "Gynaecology review if pain is severe or recurrent"]),
            "Lower abdomen / pelvis")
        add(("Ectopic pregnancy (if pregnancy is possible)", "S",
             "Sudden one-sided lower abdominal pain, with or without bleeding, when pregnancy is possible.",
             ["Urine/serum hCG", "Transvaginal ultrasound"], ["Emergency evaluation if pregnancy is possible"]),
            "Lower abdomen / pelvis")
    if p["gender"] == "Male" and gset & {"pelvis", "abdomen_lower"}:
        add(("Prostatitis / testicular problem (e.g., torsion)", "L",
             "Groin/pelvic pain with urinary symptoms or testicular pain; sudden testicular pain is an emergency.",
             ["Urinalysis", "Scrotal ultrasound", "PSA only on doctor's advice"],
             ["Sudden testicular pain → go to ER", "Otherwise urology review"]), "Pelvis / groin")
    if p["age"] >= 50 and gset & {"chest", "shoulder", "back_upper", "abdomen_upper"}:
        add(("Heart disease risk (age ≥ 50)", "S",
             "Heart problems can present as shoulder, back, jaw or upper-abdominal discomfort.",
             ["ECG", "Lipid profile", "Blood sugar / HbA1c", "Blood pressure"],
             ["Heart-healthy diet", "Regular exercise", "Medical check-up"]), "Age-related risk")

    for s in p["assoc"]:
        if s in SYMPTOM_KB:
            add(SYMPTOM_KB[s], f"Associated symptom: {s}")

    order = {"S": 0, "C": 1, "L": 2}
    causes.sort(key=lambda c: order[c["like"]])
    return causes


GENERAL_TIPS = [
    "💧 Drink enough water (about 2–3 litres/day unless advised otherwise).",
    "😴 Aim for 7–9 hours of regular, quality sleep.",
    "🥗 Eat a balanced diet rich in vegetables, fruit, whole grains and protein.",
    "🚶 Include 30 minutes of moderate activity most days (as pain allows).",
    "🧘 Manage stress with breathing exercises, meditation or yoga.",
    "🚭 Avoid smoking and limit alcohol.",
    "📝 Note when symptoms start, what worsens/relieves them – it helps your doctor.",
]


def build_ai_report(p, key):
    import anthropic  # imported lazily so the app works without it

    client = anthropic.Anthropic(api_key=key)
    system = (
        "You are a careful health-information assistant. You do NOT diagnose. "
        "Given a patient's profile, produce a concise, well-structured markdown report with these sections:\n"
        "## Summary\n## Possible causes (ordered by likelihood, include any serious cause that must be ruled out)\n"
        "## Tests a doctor may order (map each test to the cause it checks)\n"
        "## Treatments & wellness tips (self-care, lifestyle, when OTC options may help – no prescription doses)\n"
        "## When to seek care urgently\n"
        "Use plain language, avoid alarmism, and end with a one-line disclaimer."
    )
    user = (
        f"Name: {p['name']}\nAge: {p['age']}\nGender: {p['gender']}\n"
        f"Body areas with pain/discomfort: {', '.join(p['regions']) or 'none selected'}\n"
        f"Duration: {p['duration']}\nSeverity (1-10): {p['severity']}\n"
        f"Associated symptoms: {', '.join(p['assoc']) or 'none'}\n"
        f"Existing conditions / medications: {p['history'] or 'not provided'}\n"
        f"Description: {p['description'] or 'not provided'}"
    )
    msg = client.messages.create(model=AI_MODEL, max_tokens=2200, system=system,
                                 messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in msg.content if getattr(b, "text", None))


# ---------------------------------------------------------------------------
# 4. HOSPITAL FINDER (OpenStreetMap – free, no key)
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


@st.cache_data(ttl=3600, show_spinner=False)
def geocode(place):
    r = requests.get("https://nominatim.openstreetmap.org/search",
                     params={"q": place, "format": "json", "limit": 1}, headers=UA, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    return float(data[0]["lat"]), float(data[0]["lon"]), data[0]["display_name"]


@st.cache_data(ttl=3600, show_spinner=False)
def find_hospitals(lat, lon):
    hospitals = []
    for radius in (5000, 15000, 40000):
        q = f"""[out:json][timeout:30];
        (node["amenity"="hospital"](around:{radius},{lat},{lon});
         way["amenity"="hospital"](around:{radius},{lat},{lon});
         relation["amenity"="hospital"](around:{radius},{lat},{lon}););
        out center tags;"""
        r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q}, headers=UA, timeout=60)
        r.raise_for_status()
        hospitals, seen = [], set()
        for el in r.json().get("elements", []):
            t = el.get("tags", {})
            la = el.get("lat") or el.get("center", {}).get("lat")
            lo = el.get("lon") or el.get("center", {}).get("lon")
            if la is None or lo is None:
                continue
            name = t.get("name") or t.get("name:en") or "Hospital (name not listed)"
            sig = (name.lower(), round(la, 3), round(lo, 3))
            if sig in seen:
                continue
            seen.add(sig)
            addr = t.get("addr:full") or ", ".join(
                x for x in [t.get("addr:housenumber"), t.get("addr:street"), t.get("addr:suburb"), t.get("addr:city")] if x)
            hospitals.append(dict(
                name=name, lat=la, lon=lo, dist=haversine(lat, lon, la, lo), address=addr,
                phone=t.get("phone") or t.get("contact:phone") or "",
                emergency=t.get("emergency", ""), website=t.get("website") or t.get("contact:website") or "",
            ))
        if len(hospitals) >= 4:
            break
    hospitals.sort(key=lambda h: h["dist"])
    return hospitals[:10]


def hospital_section(key_prefix):
    st.subheader("🏥 Find nearby hospitals")
    st.caption("Enter your city / town (add area or country for accuracy, e.g. 'Dwarka, Delhi').")
    c1, c2 = st.columns([3, 1])
    place = c1.text_input("Your city / town", key=f"{key_prefix}_place", placeholder="e.g. Pune, India")
    go_btn = c2.button("Find hospitals", key=f"{key_prefix}_btn", type="primary")
    if go_btn and place.strip():
        try:
            with st.spinner("Locating you and searching for hospitals…"):
                loc = geocode(place.strip())
                if not loc:
                    st.session_state["hosp"] = {"error": "Couldn't find that place. Try adding the country."}
                else:
                    st.session_state["hosp"] = {"loc": loc, "list": find_hospitals(loc[0], loc[1])}
        except Exception as e:  # network / API issues
            st.session_state["hosp"] = {"error": f"Hospital search failed ({e}). Please call your local emergency number."}
    res = st.session_state.get("hosp")
    if not res:
        return
    if "error" in res:
        st.error(res["error"])
        return
    lat, lon, label = res["loc"]
    hs = res["list"]
    st.success(f"Showing results near: {label}")
    if not hs:
        st.warning("No hospitals found in the map data nearby. Call emergency services or ask someone local.")
        return
    df = pd.DataFrame([{"lat": h["lat"], "lon": h["lon"]} for h in hs] + [{"lat": lat, "lon": lon}])
    st.map(df, zoom=11)
    for i, h in enumerate(hs, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {h['name']}** — {h['dist']:.1f} km away"
                        + ("  🚑 *has emergency dept.*" if h["emergency"] == "yes" else ""))
            bits = []
            if h["address"]:
                bits.append(f"📍 {h['address']}")
            if h["phone"]:
                bits.append(f"📞 {h['phone']}")
            if bits:
                st.write("  \n".join(bits))
            links = f"[🧭 Directions](https://www.google.com/maps/dir/?api=1&destination={h['lat']},{h['lon']})"
            if h["website"]:
                links += f" · [🌐 Website]({h['website']})"
            st.markdown(links)
    st.caption("Data © OpenStreetMap contributors. Availability of services isn't guaranteed – call ahead if you can.")


# ---------------------------------------------------------------------------
# 5. UI
# ---------------------------------------------------------------------------
st.session_state.setdefault("regions", [])
st.session_state.setdefault("chart_n", 0)
st.session_state.setdefault("report", None)

with st.sidebar:
    st.header("⚙️ Settings")
    default_key = os.getenv("ANTHROPIC_API_KEY", "")
    try:
        default_key = default_key or st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        pass
    api_key = st.text_input("Anthropic API key (optional)", value=default_key, type="password",
                            help="Adds a personalised AI-written report. Without it, the built-in knowledge base is used.")
    use_ai = st.toggle("Use AI-generated report", value=bool(api_key), disabled=not api_key)
    st.divider()
    st.markdown("**Emergency numbers**  \n🇮🇳 India: **112** (national) · **108** (ambulance)  \n🇺🇸 US: **911** · 🇬🇧 UK: **999** · 🇪🇺 EU: **112**")
    st.divider()
    st.caption("⚠️ This tool provides general health information only and is **not** a substitute for professional medical advice, diagnosis or treatment.")

st.title("🩺 Healthcare Assistant")
st.caption("Tell me about yourself and your symptoms, mark the affected areas on the body, and get an informational summary.")

# ---- Step 1: basic info
st.header("1️⃣ Basic information")
c1, c2, c3 = st.columns([2, 1, 1])
name = c1.text_input("Name", key="name", placeholder="Your name")
age = c2.number_input("Age", min_value=0, max_value=120, value=30, step=1, key="age")
gender = c3.selectbox("Gender", ["Female", "Male", "Non-binary / Other", "Prefer not to say"], key="gender")

# ---- Step 2: body diagram
st.header("2️⃣ Where does it hurt?")
st.caption("Click any point on the body diagram below to mark or unmark affected areas. "
           "Selected areas will light up in red with labels.")
left, right = st.columns([1, 1])
with left:
    view = st.radio("View Perspective", ["Front", "Back"], horizontal=True, key="view")
    chart_key = f"body_{view}_{st.session_state.chart_n}"
    event = show_chart(body_figure(view, set(st.session_state.regions)), chart_key)
    clicked = clicked_names(event, view)
    if clicked:
        for n in clicked:
            if n in st.session_state.regions:
                st.session_state.regions.remove(n)
            else:
                st.session_state.regions.append(n)
        st.session_state.chart_n += 1  # reset chart selection state
        st.rerun()
    if event is None:
        st.warning("Your Streamlit version doesn't support interactive point selection. "
                   "Update Streamlit (`pip install -U streamlit`) or use the multi-select box on the right.")

with right:
    st.subheader("Selected Body Regions")
    st.caption("Interact with the map or manage selected body parts directly in this list:")
    chosen = st.multiselect("Body areas", ALL_REGIONS, default=st.session_state.regions,
                            key=f"ms_{st.session_state.chart_n}", label_visibility="collapsed",
                            placeholder="Select regions...")
    if set(chosen) != set(st.session_state.regions):
        st.session_state.regions = list(chosen)
        st.session_state.chart_n += 1
        st.rerun()
        
    if st.session_state.regions:
        st.info(f"📍 **{len(st.session_state.regions)}** region(s) selected.")
        if st.button("🗑️ Clear All Regions"):
            st.session_state.regions = []
            st.session_state.chart_n += 1
            st.rerun()

    # ---- Step 3: symptoms
    st.header("3️⃣ Current symptoms")
    description = st.text_area("Describe how you feel (type of pain, what triggers it, what helps…)",
                               placeholder="e.g. Dull ache in the lower back since yesterday, worse when bending…",
                               height=110)
    s1, s2 = st.columns(2)
    duration = s1.selectbox("How long?", ["Less than 24 hours", "1–3 days", "4–7 days", "1–4 weeks", "More than a month"])
    severity = s2.slider("Severity (1 = mild, 10 = unbearable)", 1, 10, 4)
    assoc = st.multiselect(
        "Other symptoms you have",
        ["Fever", "Fatigue", "Dizziness", "Nausea / vomiting", "Cough", "Rash", "Swelling", "Numbness / tingling",
         "Sweating", "Shortness of breath", "Chest pain / pressure", "Sudden weakness / numbness on one side",
         "Confusion / slurred speech", "Fainting / loss of consciousness", "Severe bleeding", "Coughing up blood",
         "Sudden severe headache", "Loss of bladder/bowel control"])
    history = st.text_input("Existing conditions / medicines (optional)", placeholder="e.g. diabetes, hypertension, metformin")
    self_flag = st.checkbox("🚨 I think this is an emergency")

    analyze = st.button("🔍 Analyze my symptoms", type="primary")

if analyze:
    if not st.session_state.regions and not description.strip() and not assoc:
        st.warning("Please select at least one body area, describe your symptoms, or pick an associated symptom.")
    else:
        profile = dict(name=name.strip() or "Friend", age=int(age), gender=gender,
                       regions=list(st.session_state.regions), description=description, duration=duration,
                       severity=severity, assoc=assoc, history=history, self_flag=self_flag)
        emergency = detect_emergency(profile)
        report = dict(profile=profile, emergency=emergency, causes=rule_based_causes(profile), ai=None, ai_error=None)
        if use_ai and api_key:
            try:
                with st.spinner("Generating personalised AI report…"):
                    report["ai"] = build_ai_report(profile, api_key)
            except Exception as e:
                report["ai_error"] = str(e)
        st.session_state.report = report

# ---- Results
rep = st.session_state.report
if rep:
    p = rep["profile"]
    st.divider()
    st.header(f"📋 Health summary for {p['name']}")

    if rep["emergency"]:
        st.error("### 🚨 This may be a medical emergency\n"
                 "**Call your local emergency number now (India: 112 / 108; US: 911; UK: 999) "
                 "or go to the nearest emergency department. Do not drive yourself.**\n\n"
                 + "\n".join(f"- {r}" for r in rep["emergency"]))
        if any("suicid" in w or "self-harm" in w or "kill myself" in w
               for w in RED_FLAG_WORDS if w in p["description"].lower()):
            st.warning("If you are having thoughts of harming yourself, please reach out right now to a crisis service "
                       "(India: Tele-MANAS 14416) or emergency services. You are not alone.")
        hospital_section("emg")

    # Profile recap
    with st.container(border=True):
        a, b, c = st.columns(3)
        a.metric("Age / Gender", f"{p['age']} · {p['gender']}")
        b.metric("Duration", p["duration"])
        c.metric("Severity", f"{p['severity']}/10")
        st.write("**Areas:** " + (", ".join(p["regions"]) or "—"))
        st.write("**Other symptoms:** " + (", ".join(p["assoc"]) or "—"))
        if p["description"]:
            st.write("**Your description:** " + p["description"])

    if rep["ai_error"]:
        st.warning(f"AI report unavailable ({rep['ai_error']}). Showing the built-in knowledge-base report instead.")

    if rep["ai"]:
        st.markdown(rep["ai"])
        st.caption("Below: the structured knowledge-base view of the same data.")
    # Rule-based report (always available as the structured view)
    tab_c, tab_t, tab_w = st.tabs(["🔎 Possible causes", "🧪 Tests to confirm", "🌿 Treatments & wellness tips"])

    with tab_c:
        st.caption("Ordered with serious conditions first (to rule out), then common and less common causes. "
                   "This is not a diagnosis.")
        for c in rep["causes"]:
            with st.expander(f"{LIKELIHOOD[c['like']]} — {c['name']}", expanded=c["like"] == "S"):
                st.write(c["why"])
                st.caption(f"Linked to: {c['source']}")

    with tab_t:
        rows = [{"Possible cause": c["name"], "Tests / investigations": " • ".join(c["tests"])} for c in rep["causes"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("A doctor will choose tests based on your examination; not all are needed.")

    with tab_w:
        st.subheader("Cause-specific self-care")
        for c in rep["causes"]:
            if c["like"] == "S":
                continue
            st.markdown(f"**{c['name']}**")
            for t in c["care"]:
                st.markdown(f"- {t}")
        st.subheader("General wellness tips")
        for t in GENERAL_TIPS:
            st.markdown(f"- {t}")
        st.info("OTC medicines: follow the label, check for allergies/interactions, and ask a pharmacist or doctor "
                "if you have other conditions, are pregnant, or are giving them to a child.")

    st.subheader("🩺 When to see a doctor")
    st.markdown(
        "- Symptoms last more than a few days or keep getting worse\n"
        "- Severe pain, high fever, or symptoms that interfere with daily life\n"
        "- Any new weakness, numbness, vision change, confusion, or difficulty breathing → **emergency**\n"
        "- You are pregnant, elderly, immunocompromised, or have chronic illness and feel unwell")

    if not rep["emergency"]:
        with st.expander("🏥 Need a hospital or clinic nearby anyway?"):
            hospital_section("opt")

    st.caption("⚠️ Informational only – not a medical diagnosis. Always consult a qualified healthcare professional.")