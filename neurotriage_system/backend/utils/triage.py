"""
Triage function to decide which models to run first based on patient data
"""

from typing import Dict, Any, List, Optional
import json

def triage_patient_data(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze patient data and determine which models to run first
    
    Args:
        patient_data: Dictionary containing patient information:
            - demographics: name, age, gender
            - clinical_context: imaging_type, symptoms (list), history, notes
            
    Returns:
        Dictionary with:
            - models_to_run: List of model names in priority order
            - priority: High/Medium/Low based on symptoms
            - reasoning: Explanation of triage decision
    """
    # Extract clinical context
    clinical = patient_data.get('clinical_context', {})
    symptoms = clinical.get('symptoms', [])
    
    # Parse symptoms if it's a JSON string
    if isinstance(symptoms, str):
        try:
            symptoms = json.loads(symptoms)
        except (json.JSONDecodeError, TypeError):
            symptoms = []
    
    if not isinstance(symptoms, list):
        symptoms = []
    
    history = clinical.get('history', '').lower() if clinical.get('history') else ''
    notes = clinical.get('notes', '').lower() if clinical.get('notes') else ''
    imaging_type = clinical.get('imaging_type', '').upper()
    
    # Initialize model priority
    tumor_priority = 0
    hemorrhage_priority = 0
    
    # Hemorrhage indicators
    hemorrhage_keywords = [
        'trauma', 'head injury', 'fall', 'accident', 'loss of consciousness',
        'acute', 'sudden', 'emergency'
    ]
    hemorrhage_symptoms = [
        'loss of consciousness', 'trauma history', 'nausea/vomiting',
        'headache', 'seizures'
    ]
    
    # Tumor indicators
    tumor_keywords = [
        'gradual', 'progressive', 'weakness', 'speech', 'vision',
        'cognitive', 'memory', 'chronic'
    ]
    tumor_symptoms = [
        'speech difficulties', 'motor/balance issues', 'vision problems',
        'cognitive changes', 'memory problems', 'weakness/numbness',
        'focal neurological deficits'
    ]
    
    # Analyze symptoms for hemorrhage indicators
    for symptom in symptoms:
        if any(hs in symptom.lower() for hs in hemorrhage_symptoms):
            hemorrhage_priority += 3
    
    # Analyze symptoms for tumor indicators
    for symptom in symptoms:
        if any(ts in symptom.lower() for ts in tumor_symptoms):
            tumor_priority += 2
    
    # Analyze history text for keywords
    for keyword in hemorrhage_keywords:
        if keyword in history or keyword in notes:
            hemorrhage_priority += 2
    
    for keyword in tumor_keywords:
        if keyword in history or keyword in notes:
            tumor_priority += 1
    
    # Imaging type preference
    # CT is better for acute hemorrhage detection
    if imaging_type == 'CT':
        hemorrhage_priority += 1
    # MRI is better for tumor detection
    elif imaging_type == 'MRI':
        tumor_priority += 1
    
    # Determine priority level
    total_priority = max(tumor_priority, hemorrhage_priority)
    if total_priority >= 5:
        priority_level = "High"
    elif total_priority >= 2:
        priority_level = "Medium"
    else:
        priority_level = "Low"
    
    # Decide which models to run and in what order
    models_to_run = []
    reasoning = []
    
    if hemorrhage_priority > tumor_priority:
        models_to_run = ["hemorrhage", "tumor"]
        reasoning.append(f"Hemorrhage detection prioritized (score: {hemorrhage_priority})")
        reasoning.append(f"- Acute symptoms and trauma indicators detected")
        reasoning.append(f"- Tumor detection will run as secondary check")
    elif tumor_priority > hemorrhage_priority:
        models_to_run = ["tumor", "hemorrhage"]
        reasoning.append(f"Tumor detection prioritized (score: {tumor_priority})")
        reasoning.append(f"- Progressive/neurological symptoms detected")
        reasoning.append(f"- Hemorrhage detection will run as secondary check")
    else:
        # Equal priority - run both, default to hemorrhage first for acute cases
        models_to_run = ["hemorrhage", "tumor"]
        reasoning.append("Equal priority - running hemorrhage detection first")
        reasoning.append("- Acute conditions require immediate attention")
        reasoning.append("- Tumor detection will follow")
    
    # If no specific indicators, default order
    if hemorrhage_priority == 0 and tumor_priority == 0:
        models_to_run = ["tumor", "hemorrhage"]
        reasoning = ["No specific indicators - defaulting to comprehensive screening"]
    
    return {
        "models_to_run": models_to_run,
        "priority": priority_level,
        "hemorrhage_score": hemorrhage_priority,
        "tumor_score": tumor_priority,
        "reasoning": reasoning,
        "recommended_order": models_to_run
    }

