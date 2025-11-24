"""
FastAPI entry point for NeuroTriage System
Main API server for brain anomaly detection
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime
import os
import uuid
import json
import heapq
from utils.triage import triage_patient_data
from models import TumorModel, HemorrhageModel, nnUNetTumorModel
import numpy as np
import asyncio
from asyncio import to_thread


# Model Registry: Maps model names to model instances
# New models can be added here without changing main logic
MODEL_REGISTRY = {
    "tumor": TumorModel(model_name="brats_tumor_model"),
    "hemorrhage": HemorrhageModel(model_name="seresnext_hemorrhage"),
    "tumor_nnunet": nnUNetTumorModel(model_name="nnunet_brats"),
}

# Initialize models at startup
print("[INFO] Initializing models...")
for name, model in MODEL_REGISTRY.items():
    try:
        if not getattr(model, 'model_loaded', False):
            model.load_model()
            print(f"[INFO] ✓ {name.capitalize()} model loaded successfully")
    except Exception as e:
        print(f"[WARNING] Failed to load {name} model: {e}")

app = FastAPI(
    title="NeuroTriage System API",
    description="API for brain anomaly detection including tumors and hemorrhages",
    version="1.0.0"
)

# Utility: make objects JSON-serializable (numpy arrays/scalars -> lists/scalars)
def _json_sanitize(obj):
    try:
        import numpy as _np
    except Exception:
        _np = None
    if _np is not None:
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, (_np.floating, _np.integer)):
            return obj.item()
    if isinstance(obj, (set,)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Progress update helper (best-effort; never raise)
def _set_progress(patient_id: str, phase: str, message: str, current: int = 0, total: int = 0) -> None:
    try:
        case = patient_queue.get_case(patient_id)
        if not case:
            return
        meta = case.get("metadata", {})
        meta["progress"] = {
            "phase": phase,
            "message": message,
            "current": int(current),
            "total": int(total),
            "updated_at": datetime.now().isoformat(),
        }
        patient_queue.update_case(patient_id, metadata=meta)
    except Exception:
        pass

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "NeuroTriage System API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/models")
async def list_models():
    """List all available models and their status"""
    models_info = {}
    for name, model in MODEL_REGISTRY.items():
        models_info[name] = {
            "name": name,
            "loaded": getattr(model, "model_loaded", False),
            "description": getattr(model, "__doc__", "").strip().split("\n")[0] if hasattr(model, "__doc__") else "",
            "type": "segmentation" if "nnunet" in name else "classification"
        }
    return {
        "models": models_info,
        "total": len(MODEL_REGISTRY)
    }

# Create data directory if it doesn't exist
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/patient_data", exist_ok=True)
os.makedirs("data/outputs", exist_ok=True)

# Mount static files for serving overlay images
app.mount("/data", StaticFiles(directory="data"), name="data")

# Urgency weights for different anomaly types (higher = more urgent)
URGENCY_WEIGHTS = {
    "hemorrhage": 2.0,     # Higher urgency - acute condition
    "tumor": 1.0,          # Lower urgency - typically less acute
    "tumor_nnunet": 1.5,   # nnU-Net segmentation - medium urgency
}

# Maximum lesion size for normalization (in mm³)
MAX_LESION_SIZE = 100000.0  # 100 cm³


class PriorityQueue:
    """
    Priority queue for managing patient cases with status tracking
    Stores cases sorted by priority score (highest first)
    """
    
    def __init__(self):
        """Initialize the priority queue"""
        self.cases: Dict[str, Dict[str, Any]] = {}  # patient_id -> case data
        self._heap: List[tuple] = []  # (negative_score, timestamp, patient_id) for heapq
    
    def add_case(
        self, 
        patient_id: str, 
        pathology: str, 
        score: float,
        status: str = "waiting",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a new case to the priority queue
        
        Args:
            patient_id: Unique patient identifier
            pathology: Type of pathology ('tumor', 'hemorrhage', etc.)
            score: Priority score (higher = more urgent)
            status: Case status ('waiting', 'running', 'complete')
            metadata: Additional metadata dictionary
            
        Returns:
            True if added successfully, False if patient_id already exists
        """
        if patient_id in self.cases:
            # Update existing case instead of creating duplicate
            return self.update_case(patient_id, pathology=pathology, score=score, status=status, metadata=metadata)
        
        timestamp = datetime.now().isoformat()
        case_data = {
            "patient_id": patient_id,
            "pathology": pathology,
            "score": score,
            "status": status,
            "timestamp": timestamp,
            "updated_at": timestamp,
            "metadata": metadata or {}
        }
        
        self.cases[patient_id] = case_data
        # Use negative score for max-heap (heapq is min-heap)
        heapq.heappush(self._heap, (-score, timestamp, patient_id))
        
        return True
    
    def update_case(
        self,
        patient_id: str,
        pathology: Optional[str] = None,
        score: Optional[float] = None,
        status: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing case in the queue
        
        Args:
            patient_id: Patient identifier
            pathology: Updated pathology (optional)
            score: Updated priority score (optional)
            status: Updated status (optional)
            metadata: Updated metadata (optional, merged with existing)
            
        Returns:
            True if updated, False if patient_id not found
        """
        if patient_id not in self.cases:
            return False
        
        case = self.cases[patient_id]
        
        # Update fields
        if pathology is not None:
            case["pathology"] = pathology
        if score is not None:
            old_score = case["score"]
            case["score"] = score
            # Rebuild heap if score changed
            if score != old_score:
                self._rebuild_heap()
        if status is not None:
            case["status"] = status
        if metadata is not None:
            case["metadata"].update(metadata)
        
        case["updated_at"] = datetime.now().isoformat()
        
        return True
    
    def update_case_status(self, patient_id: str, status: str) -> bool:
        """
        Update the status of a case
        
        Args:
            patient_id: Patient identifier
            status: New status ('waiting', 'running', 'complete')
            
        Returns:
            True if updated, False if patient_id not found
        """
        return self.update_case(patient_id, status=status)
    
    def get_next_case(self, status_filter: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the next case with highest priority score
        
        Args:
            status_filter: Only return cases with this status ('waiting', 'running', 'complete')
                          If None, returns highest priority case regardless of status
                          
        Returns:
            Case dictionary or None if queue is empty or no matching cases
        """
        if not self.cases:
            return None
        
        # Get all cases sorted by score
        sorted_cases = self.get_all_cases(sorted_by="score", reverse=True)
        
        if status_filter:
            # Filter by status
            filtered = [case for case in sorted_cases if case["status"] == status_filter]
            return filtered[0] if filtered else None
        else:
            # Return highest priority case
            return sorted_cases[0]
    
    def get_case(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific case by patient_id
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Case dictionary or None if not found
        """
        return self.cases.get(patient_id)
    
    def get_all_cases(self, sorted_by: str = "score", reverse: bool = True) -> List[Dict[str, Any]]:
        """
        Get all cases, optionally sorted
        
        Args:
            sorted_by: Field to sort by ('score', 'timestamp', 'updated_at')
            reverse: Sort in descending order
            
        Returns:
            List of all cases
        """
        cases = list(self.cases.values())
        cases.sort(key=lambda x: x.get(sorted_by, 0), reverse=reverse)
        return cases
    
    def remove_case(self, patient_id: str) -> bool:
        """
        Remove a case from the queue
        
        Args:
            patient_id: Patient identifier to remove
            
        Returns:
            True if removed, False if not found
        """
        if patient_id not in self.cases:
            return False
        
        del self.cases[patient_id]
        self._rebuild_heap()
        return True
    
    def _rebuild_heap(self) -> None:
        """Rebuild the heap structure after score updates"""
        self._heap.clear()
        for patient_id, case in self.cases.items():
            heapq.heappush(self._heap, (-case["score"], case["timestamp"], patient_id))
    
    def get_status_counts(self) -> Dict[str, int]:
        """
        Get count of cases by status
        
        Returns:
            Dictionary mapping status to count
        """
        counts = {"waiting": 0, "running": 0, "complete": 0}
        for case in self.cases.values():
            status = case.get("status", "waiting")
            if status in counts:
                counts[status] += 1
            else:
                counts[status] = 1
        return counts
    
    def size(self) -> int:
        """Get total number of cases in queue"""
        return len(self.cases)


# Initialize global priority queue instance
patient_queue = PriorityQueue()


def compute_priority_score(prediction: Dict[str, Any], model_type: str) -> float:
    """
    Compute priority score for patient case
    
    Formula: score = model_confidence * (lesion_size or probability) * urgency_weight
    
    Args:
        prediction: Model prediction dictionary containing:
            - For segmentation (tumor): 'probability', 'volumes' with 'whole_tumor_volume'
            - For classification (hemorrhage): 'probability', 'confidence'
        model_type: Type of model ('tumor' or 'hemorrhage')
        
    Returns:
        Priority score (higher = more urgent)
    """
    # Get urgency weight based on model type
    urgency_weight = URGENCY_WEIGHTS.get(model_type, 1.0)
    
    # Get model confidence (inverse of uncertainty, or use probability as proxy)
    uncertainty = prediction.get("uncertainty", 0.0)
    model_confidence = 1.0 - min(uncertainty, 1.0)  # Convert uncertainty to confidence
    
    # If uncertainty not available, use probability as confidence proxy
    if model_confidence <= 0 or uncertainty == 0:
        model_confidence = prediction.get("probability", prediction.get("confidence", 0.5))
    
    # Get lesion_size or probability
    if model_type == "tumor":
        # For segmentation: use lesion size (volume in mm³)
        volumes = prediction.get("volumes", {})
        lesion_size = volumes.get("whole_tumor_volume", 0.0)
        
        # Normalize lesion size to 0-1 range (using max expected size)
        normalized_lesion_size = min(lesion_size / MAX_LESION_SIZE, 1.0)
        
        # Use normalized lesion size, fallback to probability if volume is 0
        if normalized_lesion_size > 0:
            severity_factor = normalized_lesion_size
        else:
            severity_factor = prediction.get("probability", 0.0)
    else:
        # For classification: use probability directly
        severity_factor = prediction.get("probability", prediction.get("confidence", 0.0))
    
    # Calculate priority score
    priority_score = model_confidence * severity_factor * urgency_weight
    
    return priority_score


def add_to_queue(patient_id: str, prediction: Dict[str, Any], model_type: str, 
                 patient_data: Dict[str, Any], status: str = "waiting") -> None:
    """
    Add patient case to priority queue (wrapper for PriorityQueue.add_case)
    
    Args:
        patient_id: Unique patient identifier
        prediction: Model prediction dictionary
        model_type: Type of model ('tumor' or 'hemorrhage')
        patient_data: Full patient data dictionary
        status: Initial status (default: 'waiting')
    """
    # Compute priority score
    priority_score = compute_priority_score(prediction, model_type)
    
    # Add to queue with metadata
    metadata = {
        "prediction": prediction,
        "model_type": model_type,
        "patient_data": patient_data,
    }
    
    patient_queue.add_case(
        patient_id=patient_id,
        pathology=model_type,
        score=priority_score,
        status=status,
        metadata=metadata
    )


def get_queue(status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get sorted patient queue (descending by priority score)
    
    Args:
        status_filter: Optional filter by status ('waiting', 'running', 'complete')
        
    Returns:
        List of patient cases sorted by priority (highest first)
    """
    if status_filter:
        cases = [case for case in patient_queue.get_all_cases() if case["status"] == status_filter]
    else:
        cases = patient_queue.get_all_cases()
    
    # Sort by score (descending)
    cases.sort(key=lambda x: x["score"], reverse=True)
    return cases


def remove_from_queue(patient_id: str) -> bool:
    """
    Remove patient case from queue (wrapper for PriorityQueue.remove_case)
    
    Args:
        patient_id: Patient identifier to remove
        
    Returns:
        True if removed, False if not found
    """
    return patient_queue.remove_case(patient_id)

@app.post("/upload_patient_data")
async def upload_patient_data(
    patient_data_json: str = Form(...),  # JSON string with patient info
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    brain_files: Optional[List[UploadFile]] = File(None),
    bone_files: Optional[List[UploadFile]] = File(None),
    mri_files: Optional[List[UploadFile]] = File(None)
):
    """
    Upload patient data including demographics, clinical context, and imaging file
    
    Accepts multipart/form-data with:
    - patient_data_json: JSON string containing patient information
    - file: DICOM or NIfTI imaging file
    
    JSON structure:
    {
        "demographics": {
            "name": str,
            "age": int,
            "gender": str
        },
        "clinical_context": {
            "imaging_type": "MRI" or "CT",
            "symptoms": [list of strings],
            "history": str (optional),
            "notes": str (optional)
        }
    }
    
    Returns:
        Upload confirmation with patient ID, file path, and triage results
    """
    try:
        # Parse JSON patient data
        patient_data = json.loads(patient_data_json)
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Invalid JSON format: {str(e)}"
        }
    
    # Validate required fields
    if "demographics" not in patient_data or "clinical_context" not in patient_data:
        return {
            "success": False,
            "error": "Missing required fields: 'demographics' and 'clinical_context' required"
        }
    
    # Generate unique patient ID
    patient_id = str(uuid.uuid4())
    
    # Extract scan type from patient data
    scan_type = patient_data.get('scan_type', 'CT')

    # Check for brain/bone folder mode (CT scans)
    has_brain_bone = (brain_files and len(brain_files) > 0 and 
                     bone_files and len(bone_files) > 0)
    
    # Check for MRI folder mode
    has_mri = (mri_files and len(mri_files) > 0)
    
    # Normalize to a list of UploadFile
    upload_list: List[UploadFile] = []
    if has_brain_bone:
        # Validate brain and bone file counts match
        if len(brain_files) != len(bone_files):
            return {"success": False, "error": f"Brain and bone file counts must match. Got {len(brain_files)} brain, {len(bone_files)} bone."}
        upload_list = []  # Will handle brain/bone separately
    elif has_mri:
        upload_list = []  # Will handle MRI separately
    elif files and isinstance(files, list) and len(files) > 0:
        upload_list = files
    elif file is not None:
        upload_list = [file]
    else:
        return {"success": False, "error": "No file(s) provided"}

    allowed_extensions = ['dcm', 'dicom', 'nii', 'nifti', 'nii.gz', 'nifti.gz', 'jpg', 'jpeg', 'png']

    # Determine pathology based on scan type
    pathology_type = "tumor" if scan_type == "MRI" else "hemorrhage"
    model_type = "tumor" if scan_type == "MRI" else "hemorrhage"
    
    # Create a provisional queue entry so progress is visible during upload/conversion
    try:
        patient_queue.add_case(
            patient_id=patient_id,
            pathology=pathology_type,
            score=0.0,
            status="uploading",
            metadata={
                "prediction": {},
                "model_type": model_type,
                "patient_data": patient_data,
                "progress": {
                    "phase": "uploading",
                    "message": "Preparing upload",
                    "current": 0,
                    "total": len(upload_list),
                    "updated_at": datetime.now().isoformat(),
                },
            },
        )
    except Exception:
        pass

    saved_paths: List[str] = []
    saved_names: List[str] = []
    total_size = 0
    converted_from_images = False

    # Optional: JPG/PNG to DICOM conversion (Secondary Capture)
    def _convert_image_to_dicom(img_path: str, out_path: str) -> str:
        try:
            import pydicom
            from pydicom.dataset import Dataset, FileDataset
            import datetime as _dt
            import cv2 as _cv
            img = _cv.imread(img_path, _cv.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read image: {img_path}")
            meta = Dataset()
            meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
            meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            ds = FileDataset(out_path, {}, file_meta=meta, preamble=b"\0" * 128)
            ds.Modality = "OT"
            ds.SOPClassUID = meta.MediaStorageSOPClassUID
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            now = _dt.datetime.now()
            ds.PatientName = "Converted"
            ds.PatientID = patient_id
            ds.StudyDate = now.strftime("%Y%m%d")
            ds.StudyTime = now.strftime("%H%M%S")
            ds.Rows, ds.Columns = img.shape[:2]
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 8
            ds.BitsStored = 8
            ds.HighBit = 7
            ds.PixelRepresentation = 0
            ds.PixelData = img.tobytes()
            ds.is_little_endian = True
            ds.is_implicit_VR = False
            ds.save_as(out_path)
            return out_path
        except Exception as e:
            raise RuntimeError(f"Conversion to DICOM failed for {img_path}: {e}")

    if has_brain_bone:
        # Handle brain/bone folder mode
        num_slices = len(brain_files)
        brain_paths = []
        bone_paths = []
        
        # Save brain files
        for idx, uf in enumerate(brain_files):
            _set_progress(patient_id, phase="uploading", message=f"Saving brain slice {idx+1}/{num_slices}", current=idx+1, total=num_slices*2)
            name = uf.filename
            out_path = f"data/uploads/{patient_id}_brain_{idx}_{name}"
            with open(out_path, "wb") as buffer:
                content = await uf.read()
                buffer.write(content)
                total_size += len(content)
            brain_paths.append(out_path)
        
        # Save bone files  
        for idx, uf in enumerate(bone_files):
            _set_progress(patient_id, phase="uploading", message=f"Saving bone slice {idx+1}/{num_slices}", current=num_slices+idx+1, total=num_slices*2)
            name = uf.filename
            out_path = f"data/uploads/{patient_id}_bone_{idx}_{name}"
            with open(out_path, "wb") as buffer:
                content = await uf.read()
                buffer.write(content)
                total_size += len(content)
            bone_paths.append(out_path)
        
        # Pair brain/bone paths
        for idx in range(num_slices):
            saved_paths.append(brain_paths[idx])
            saved_paths.append(bone_paths[idx])
            saved_names.append(f"slice_{idx}_brain.jpg")
            saved_names.append(f"slice_{idx}_bone.jpg")
        
        converted_from_images = True
    elif has_mri:
        # Handle MRI folder mode
        num_slices = len(mri_files)
        
        # Save MRI files
        for idx, uf in enumerate(mri_files):
            _set_progress(patient_id, phase="uploading", message=f"Saving MRI slice {idx+1}/{num_slices}", current=idx+1, total=num_slices)
            name = uf.filename
            out_path = f"data/uploads/{patient_id}_mri_{idx}_{name}"
            with open(out_path, "wb") as buffer:
                content = await uf.read()
                buffer.write(content)
                total_size += len(content)
            saved_paths.append(out_path)
            saved_names.append(f"slice_{idx}_mri.jpg")
        
        converted_from_images = True
    else:
        # Legacy mode: handle single/multi file uploads
        for idx, uf in enumerate(upload_list):
            _set_progress(patient_id, phase="uploading", message=f"Saving file {idx+1}/{len(upload_list)}", current=idx+1, total=len(upload_list))
            name = uf.filename
            lname = (name or "").lower()
            is_nii_gz = lname.endswith('.nii.gz') or lname.endswith('.nifti.gz')
            ext = 'nii.gz' if is_nii_gz else lname.split('.')[-1] if '.' in lname else ''
            if ext not in allowed_extensions:
                return {"success": False, "error": f"Invalid file type for {name}. Allowed: {allowed_extensions}"}
            out_path = f"data/uploads/{patient_id}_{idx}_{name}"
            with open(out_path, "wb") as buffer:
                content = await uf.read()
                buffer.write(content)
                total_size += len(content)
            # If image file, convert to DICOM and use the .dcm path for downstream
            if ext in ['jpg', 'jpeg', 'png']:
                _set_progress(patient_id, phase="converting", message=f"Converting to DICOM {idx+1}/{len(upload_list)}", current=idx+1, total=len(upload_list))
                dcm_out = out_path + ".dcm"
                dcm_path = _convert_image_to_dicom(out_path, dcm_out)
                saved_paths.append(dcm_path)
                converted_from_images = True
            else:
                saved_paths.append(out_path)
            saved_names.append(name)

    # Add metadata to patient data
    patient_data["patient_id"] = patient_id
    patient_data["file_info"] = {
        "filenames": saved_names,
        "file_paths": saved_paths,
        "file_count": len(saved_paths),
        "upload_timestamp": datetime.now().isoformat(),
        "total_size": total_size,
        "converted_from_images": converted_from_images,
    }
    
    # Save patient data as JSON (in a real app, save to database)
    patient_data_path = f"data/patient_data/{patient_id}.json"
    with open(patient_data_path, "w") as f:
        json.dump(patient_data, f, indent=2)
    
    # Pass JSON to triage function to decide which models to run first
    triage_result = triage_patient_data(patient_data)
    
    # Note: In production, run models and add to queue
    # For now, we'll add a placeholder prediction that can be updated after model inference
    placeholder_prediction = {
        "probability": 0.5,  # Will be replaced with actual prediction
        "uncertainty": 0.2,
        "anomaly_detected": False,
    }
    
    # Use the model_type determined from scan_type (already set earlier)
    primary_model = model_type
    
    # Compute priority score (will be updated after actual model inference)
    priority_score = compute_priority_score(placeholder_prediction, primary_model)
    
    # Store initial queue entry (will be updated when actual prediction is available)
    initial_entry = {
        "patient_id": patient_id,
        "prediction": placeholder_prediction,
        "model_type": primary_model,
        "patient_data": patient_data,
        "priority_score": priority_score,
        "status": "pending_inference",
        "triage": triage_result,
        "progress": {
            "phase": "queued",
            "message": "Queued for analysis",
            "current": 0,
            "total": len(saved_paths),
            "updated_at": datetime.now().isoformat(),
        },
    }
    
    # Add initial placeholder case to the in-memory priority queue so subsequent
    # GET /get_result/{patient_id} and queue endpoints can find it immediately.
    try:
        patient_queue.add_case(
            patient_id=patient_id,
            pathology=primary_model,
            score=priority_score,
            status="pending_inference",
            metadata=initial_entry
        )
    except Exception as e:
        print(f"Warning: failed to add initial case to queue for {patient_id}: {e}")

    # Schedule background inference so frontend gets real results automatically
    try:
        asyncio.create_task(_background_run_prediction(primary_model, patient_id))
    except Exception as e:
        print(f"Warning: could not schedule background inference for {patient_id}: {e}")

    return {
        "success": True,
        "patient_id": patient_id,
        "message": "Patient data uploaded successfully",
        "file_paths": saved_paths,
        "file_count": len(saved_paths),
        "triage": triage_result,
        "priority_score": priority_score,
        "queue_status": "added"
    }


async def _background_run_prediction(model_type: str, patient_id: str):
    """Background worker: load model, run inference, save overlay, update queue and patient JSON."""
    try:
        patient_data_path = f"data/patient_data/{patient_id}.json"
        if not os.path.exists(patient_data_path):
            print(f"_background_run_prediction: patient JSON not found: {patient_data_path}")
            return
        with open(patient_data_path, "r") as f:
            patient_data = json.load(f)
        file_info = patient_data.get("file_info", {})
        file_paths = file_info.get("file_paths")
        file_path = file_info.get("file_path")
        paths: List[str] = []
        if isinstance(file_paths, list) and len(file_paths) > 0:
            paths = [p for p in file_paths if isinstance(p, str)]
        elif isinstance(file_path, str) and len(file_path) > 0:
            paths = [file_path]
        else:
            print(f"_background_run_prediction: no file paths found in patient data for {patient_id}")
            return
        for p in paths:
            if not os.path.exists(p):
                print(f"_background_run_prediction: file not found: {p}")
                return
        model = MODEL_REGISTRY.get(model_type)
        if model is None:
            print(f"_background_run_prediction: unknown model_type: {model_type}")
            return

        # Load model on background thread if needed
        if not getattr(model, "model_loaded", False):
            _set_progress(patient_id, phase="loading_model", message="Loading model", current=0, total=len(paths))
            await to_thread(model.load_model)

        # Detect if we're in brain/bone mode (even indices brain, odd indices bone)
        is_brain_bone_mode = (len(paths) % 2 == 0 and 
                              any('_brain_' in p for p in paths[:len(paths)//2]) and
                              any('_bone_' in p for p in paths[len(paths)//2:]))
        
        _set_progress(patient_id, phase="preprocessing", message="Preprocessing images", current=0, total=len(paths))
        # Incremental preprocessing with progress updates
        arr_list = []
        
        if is_brain_bone_mode:
            # Brain/bone paired mode: process in pairs
            num_slices = len(paths) // 2
            print(f"[DEBUG] Brain/bone mode detected: {num_slices} slice pairs")
            
            for i in range(num_slices):
                brain_path = paths[i * 2]
                bone_path = paths[i * 2 + 1]
                try:
                    _set_progress(
                        patient_id,
                        phase="preprocessing",
                        message=f"Preprocessing pair {i+1}/{num_slices}",
                        current=i * 2,
                        total=len(paths),
                    )
                    # Preprocess brain+bone pair together
                    single = await to_thread(model.preprocess, None, brain_path, bone_path)
                    single_arr = single
                    if hasattr(single, "numpy"):
                        single_arr = single.numpy()
                    # Ensure shape (C, H, W)
                    if isinstance(single_arr, np.ndarray):
                        if single_arr.ndim == 4:
                            # (1, C, H, W) -> (C, H, W)
                            single_arr = single_arr[0]
                    arr_list.append(single_arr)
                    print(f"[DEBUG] Successfully preprocessed pair {i+1}: shape={single_arr.shape}")
                except Exception as _e:
                    print(f"[ERROR] Failed to preprocess pair {i+1}: {_e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    _set_progress(
                        patient_id,
                        phase="preprocessing",
                        message=f"Preprocessed pair {i+1}/{num_slices}",
                        current=(i + 1) * 2,
                        total=len(paths),
                    )
        else:
            # Legacy mode: process individually
            N_total = len(paths)
            print(f"[DEBUG] Legacy mode: {N_total} individual slices")
            
            for i, p in enumerate(paths):
                try:
                    # announce start of this item before heavy work so UI shows progress
                    _set_progress(
                        patient_id,
                        phase="preprocessing",
                        message=f"Preprocessing: {os.path.basename(p)} ({i+1}/{N_total})",
                        current=i,
                        total=N_total,
                    )
                    # preprocess single path to get one slice (shape (1, C, H, W) or (C, H, W))
                    single = await to_thread(model.preprocess, [p])
                    single_arr = single
                    if hasattr(single, "numpy"):
                        single_arr = single.numpy()
                    # Ensure shape (C, H, W)
                    if isinstance(single_arr, np.ndarray):
                        if single_arr.ndim == 4:
                            # (1, C, H, W) -> (C, H, W)
                            single_arr = single_arr[0]
                    arr_list.append(single_arr)
                    print(f"[DEBUG] Successfully preprocessed {os.path.basename(p)}: shape={single_arr.shape}")
                except Exception as _e:
                    # Skip problematic slice but continue
                    print(f"[ERROR] Failed to preprocess {os.path.basename(p)}: {_e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    _set_progress(
                        patient_id,
                        phase="preprocessing",
                        message=f"Preprocessed: {os.path.basename(p)} ({i+1}/{N_total})",
                        current=i+1,
                        total=N_total,
                    )
        # Stack into batch (N, C, H, W) - or keep as list if shapes vary
        if arr_list:
            # Check if all shapes match for stacking
            shapes = [arr.shape for arr in arr_list]
            if len(set(shapes)) == 1:
                # All same shape - can stack
                preprocessed = np.stack(arr_list, axis=0)
                print(f"[DEBUG] Stacked {len(arr_list)} slices into batch: {preprocessed.shape}")
            else:
                # Different shapes - keep as list (tumor CV detection handles this)
                preprocessed = arr_list
                print(f"[DEBUG] Keeping {len(arr_list)} variable-sized slices as list")
        else:
            # Fallback: call preprocess with all paths
            preprocessed = await to_thread(model.preprocess, paths)
        # Explicit transition marker so UI doesn't appear stuck at last preprocessing item
        try:
            _set_progress(patient_id, phase="preprocessing", message="Preprocessing complete", current=len(paths), total=len(paths))
        except Exception:
            pass
        # Per-slice prediction with progress
        try:
            # Handle both numpy array and list of arrays
            if isinstance(preprocessed, list):
                arr = preprocessed  # Keep as list
                Np = len(arr)
                print(f"[DEBUG] Starting prediction for {Np} slices (list of variable-sized arrays)")
            else:
                arr = preprocessed if isinstance(preprocessed, np.ndarray) else np.asarray(preprocessed)
                if arr.ndim == 3:
                    arr = arr[np.newaxis, ...]
                Np = arr.shape[0]
                print(f"[DEBUG] Starting prediction for {Np} slices, array shape: {arr.shape}, dtype: {arr.dtype}")
            
            _set_progress(patient_id, phase="predicting", message=f"Predicting 0/{Np}", current=0, total=Np)
            probs_sum = None  # shape (C,)
            per_probs = []
            for i in range(Np):
                try:
                    fname = None
                    if i < len(paths):
                        try:
                            fname = os.path.basename(paths[i])
                        except Exception:
                            fname = None
                    msg = f"Predicting {i+1}/{Np}" if not fname else f"Predicting: {fname} ({i+1}/{Np})"
                    _set_progress(patient_id, phase="predicting", message=msg, current=i+1, total=Np)
                    # Predict single slice
                    single = arr[i]
                    print(f"[DEBUG] Slice {i+1}/{Np} ({fname}): shape={single.shape}, dtype={single.dtype}, min={single.min():.3f}, max={single.max():.3f}")
                    # Validate shape before prediction
                    if single.ndim != 3:
                        print(f"[ERROR] Invalid slice shape {single.shape}, expected 3D (C,H,W). Skipping.")
                        continue
                    raw_i = await to_thread(model.predict, single)
                    p_i = raw_i.get("probabilities")
                    if isinstance(p_i, np.ndarray):
                        # p_i can be (1,C); squeeze to (C,)
                        if p_i.ndim == 2 and p_i.shape[0] == 1:
                            p_i = p_i[0]
                        per_probs.append(p_i)
                        probs_sum = p_i if probs_sum is None else (probs_sum + p_i)
                        print(f"[DEBUG] Slice {i+1} prediction successful: prob shape={p_i.shape}")
                    else:
                        print(f"[WARNING] Slice {i+1} returned non-array probabilities: {type(p_i)}")
                except Exception as e:
                    print(f"[ERROR] Prediction failed for slice {i+1}/{Np} ({fname}): {e}")
                    import traceback
                    traceback.print_exc()
                    _set_progress(patient_id, phase="predicting_error", message=f"Error on {fname or f'slice {i+1}'}: {str(e)[:50]}", current=i+1, total=Np)
                    # If we have partial results, break out and use them
                    if probs_sum is not None and len(per_probs) > 0:
                        print(f"[INFO] Breaking prediction loop early with {len(per_probs)} successful predictions")
                        break
                    continue
            # Aggregate
            if probs_sum is not None and len(per_probs) > 0:
                probs_avg = probs_sum / float(len(per_probs))  # (C,)
                print(f"[DEBUG] Aggregated probabilities from {len(per_probs)} slices: {probs_avg}")
                
                # Use model.predict() to get proper probability AND uncertainty
                # Don't recalculate uncertainty here - let the model do it!
                hemorrhage_prob = float(probs_avg[min(getattr(model, 'hemorrhage_class_idx', 1) if probs_avg.shape[0] > 1 else 0, probs_avg.shape[0]-1)])
                print(f"[DEBUG] Hemorrhage probability from aggregation: {hemorrhage_prob:.4f}")
                
                # Create a proper input for model.predict() to get uncertainty
                # Use the first slice as representative for uncertainty calculation
                representative_slice = arr[0] if not isinstance(arr, list) else arr[0]
                raw_pred_with_unc = await to_thread(model.predict, representative_slice)
                model_uncertainty = raw_pred_with_unc.get("uncertainty", 0.2)  # Use model's uncertainty!
                
                print(f"[DEBUG] Using model's uncertainty: {model_uncertainty:.4f}")
                
                # Build a raw_pred-like dict to feed postprocess
                raw_pred = {
                    "raw_logits": None,
                    "probabilities": probs_avg,
                    "mean_probabilities": probs_avg,
                    "hemorrhage_probability": hemorrhage_prob,
                    "uncertainty": model_uncertainty,  # Use model's uncertainty, not entropy!
                }
                prediction = await to_thread(model.postprocess, raw_pred)
                print(f"[DEBUG] Final prediction after postprocess: prob={prediction.get('probability', 0):.4f}, unc={prediction.get('uncertainty', 0):.4f}")
            else:
                # fallback to batch predict if per-slice failed globally
                raw_pred = await to_thread(model.predict, arr)
                prediction = await to_thread(model.postprocess, raw_pred)
        except Exception as e:
            print(f"Warning: prediction failed for {patient_id}: {e}")
            prediction = {"probability": 0.0, "uncertainty": 1.0, "anomaly_detected": False, "model_type": model_type}
            _set_progress(patient_id, phase="predicting_error", message="Prediction failed, continuing with overlays", current=len(paths), total=len(paths))
        # tag prediction with model_type for downstream consumers
        try:
            prediction["model_type"] = model_type
        except Exception:
            pass

        # attempt explanation/overlay for all slices/images
        overlay_path = None
        # PERFORMANCE: Grad-CAM is very slow for SE-ResNeXt-101 on CPU (~30s per slice)
        # Set to False to disable heatmaps for faster results
        # Set to True to enable visual explainability (recommended if GPU available)
        ENABLE_GRADCAM = True
        
        try:
            print(f"[DEBUG] Starting overlay generation for {patient_id} (Grad-CAM: {'enabled' if ENABLE_GRADCAM else 'disabled for speed'})")
            from utils.explainability import overlay_heatmap, save_explanation_overlay
            raw_paths = []
            overlay_paths = []
            # Determine batch dimension (N,3,H,W) or single (3,H,W) or list
            if isinstance(preprocessed, list):
                arr = preprocessed  # Keep as list
                N = len(arr)
                print(f"[DEBUG] Overlay generation for {N} variable-sized slices (list)")
            else:
                arr = preprocessed if isinstance(preprocessed, np.ndarray) else np.asarray(preprocessed)
                if arr.ndim == 3:
                    arr = arr[np.newaxis, ...]
                N = arr.shape[0]
            # announce overlays phase start
            _set_progress(patient_id, phase="generating_overlays", message=f"Generating overlays 0/{N}", current=0, total=N)
            for i in range(N):
                fname = None
                original_path = None
                try:
                    if i < len(paths):
                        fname = os.path.basename(paths[i])
                        original_path = paths[i]
                except Exception:
                    fname = None
                msg = f"Generating overlays {i+1}/{N}" if not fname else f"Generating overlays: {fname} ({i+1}/{N})"
                _set_progress(patient_id, phase="generating_overlays", message=msg, current=i+1, total=N)
                
                # Load ORIGINAL image for display (not preprocessed 224x224)
                if original_path and hasattr(model, 'load_original_image'):
                    try:
                        display_slice = model.load_original_image(original_path)
                        print(f"[DEBUG] Loaded original image for display: {display_slice.shape}")
                    except Exception as e:
                        print(f"[WARNING] Failed to load original image: {e}, using preprocessed")
                        image_slice = arr[i]  # (3, 224, 224) from preprocessing
                        display_slice = model.convert_to_display_image(image_slice) if hasattr(model, 'convert_to_display_image') else image_slice
                else:
                    # Fallback: use preprocessed
                    image_slice = arr[i]  # (3, 224, 224) from preprocessing
                    if hasattr(model, 'convert_to_display_image'):
                        display_slice = model.convert_to_display_image(image_slice)
                        print(f"[DEBUG] Converted to display format: {display_slice.shape}")
                    else:
                        display_slice = image_slice
                
                # Save ORIGINAL image (full resolution, not 224x224)
                raw_filename = f"{model_type}_raw_{patient_id}_{i}.png"
                raw_path = save_explanation_overlay(display_slice, "data/outputs", raw_filename)
                print(f"[DEBUG] Saved original image: {raw_path}")
                raw_paths.append(raw_path)
                
                # Try to generate heatmap if enabled
                heatmap = None
                if ENABLE_GRADCAM:
                    max_prob = float(prediction.get("probability", 0.5))
                    uncertainty = float(prediction.get("uncertainty", 0.0))
                    print(f"[DEBUG] Generating heatmap for slice {i} (probability: {max_prob:.2f}, uncertainty: {uncertainty:.2f})")
                    
                    # Warn if model is highly uncertain
                    if uncertainty > 0.8:
                        print(f"[WARNING] Model is highly uncertain ({uncertainty:.1%}) - heatmap may be diffuse/weak")
                    
                    # Try per-slice heatmap (use preprocessed slice for model)
                    try:
                        model_slice = arr[i]  # Get preprocessed (3, 512, 512) for model
                        heatmap = await to_thread(model.explain, model_slice)
                        if heatmap is not None:
                            print(f"[DEBUG] Generated heatmap: shape={heatmap.shape}, range=[{heatmap.min():.3f}, {heatmap.max():.3f}]")
                        else:
                            print(f"[DEBUG] Heatmap generation returned None")
                    except Exception as e:
                        print(f"[ERROR] Heatmap generation failed: {e}")
                        import traceback
                        traceback.print_exc()
                        heatmap = None
                else:
                    print(f"[DEBUG] Grad-CAM disabled for speed, skipping heatmap for slice {i}")
                
                if heatmap is None:
                    # No heatmap available, just use original image as overlay
                    print(f"[WARNING] No heatmap available for slice {i}, using raw image")
                    filename = f"{model_type}_overlay_{patient_id}_{i}.png"
                    opath = save_explanation_overlay(display_slice, "data/outputs", filename)
                    overlay_paths.append(opath)
                    if overlay_path is None:
                        overlay_path = opath
                else:
                    # Overlay heatmap on ORIGINAL image (full resolution)
                    heatmap_array = np.asarray(heatmap, dtype=np.float32)
                    print(f"[DEBUG] Overlaying heatmap on original image: orig={display_slice.shape}, heatmap={heatmap_array.shape}")
                    overlay_img = overlay_heatmap(display_slice, heatmap_array, alpha=0.45)
                    filename = f"{model_type}_overlay_{patient_id}_{i}.png"
                    opath = save_explanation_overlay(overlay_img, "data/outputs", filename)
                    print(f"[DEBUG] Saved overlay image with heatmap: {opath}")
                    overlay_paths.append(opath)
                    if overlay_path is None:
                        overlay_path = opath
            # Backward-compatible single entries
            if raw_paths:
                prediction["raw_image_path"] = raw_paths[0]
            if overlay_paths:
                prediction["overlay_image_path"] = overlay_paths[0]
            prediction["raw_image_paths"] = raw_paths
            prediction["overlay_image_paths"] = overlay_paths
            print(f"[DEBUG] Generated {len(raw_paths)} raw images and {len(overlay_paths)} overlays")
            print(f"[DEBUG] Raw paths: {raw_paths}")
            print(f"[DEBUG] Overlay paths: {overlay_paths}")
        except Exception as e:
            print(f"[ERROR] Could not generate overlay/explanation for {patient_id}: {e}")
            import traceback
            traceback.print_exc()

        # update queue
        try:
            priority_score = compute_priority_score(prediction, model_type)
            if patient_queue.get_case(patient_id):
                patient_queue.update_case(
                    patient_id=patient_id,
                    score=priority_score,
                    status="complete",
                    metadata={"prediction": prediction, "model_type": model_type, "patient_data": patient_data, "overlay_path": overlay_path, "progress": {"phase": "complete", "message": "Analysis complete", "current": len(prediction.get("raw_image_paths", [])), "total": len(prediction.get("raw_image_paths", [])), "updated_at": datetime.now().isoformat()}}
                )
            else:
                patient_queue.add_case(
                    patient_id=patient_id,
                    pathology=model_type,
                    score=priority_score,
                    status="complete",
                    metadata={"prediction": prediction, "model_type": model_type, "patient_data": patient_data, "overlay_path": overlay_path, "progress": {"phase": "complete", "message": "Analysis complete", "current": len(prediction.get("raw_image_paths", [])), "total": len(prediction.get("raw_image_paths", [])), "updated_at": datetime.now().isoformat()}}
                )
        except Exception as e:
            print(f"Warning: could not update queue for {patient_id}: {e}")
            # Ensure progress shows completion even if queue update failed
            _set_progress(patient_id, phase="complete", message="Analysis complete", current=len(prediction.get("raw_image_paths", []) or []), total=len(prediction.get("raw_image_paths", []) or []))

        # persist prediction in JSON
        try:
            patient_data["prediction"] = prediction
            patient_data["last_prediction_time"] = datetime.now().isoformat()
            with open(patient_data_path, "w") as f:
                json.dump(patient_data, f, indent=2, default=_json_sanitize)
        except Exception as e:
            print(f"Warning: could not save prediction into patient JSON for {patient_id}: {e}")
    except Exception as e:
        print(f"Error in background prediction for {patient_id}: {e}")


@app.post("/predict/{model_type}")
async def run_prediction(
    model_type: str,
    patient_id: str = Form(...),
    update_queue: bool = Form(True)
):
    """
    Run model prediction and optionally update patient queue
    
    Args:
        model_type: Type of model ('tumor' or 'hemorrhage')
        patient_id: Patient identifier
        update_queue: Whether to add/update in queue after prediction
        
    Returns:
        Prediction results and updated priority score
    """
    if model_type not in MODEL_REGISTRY:
        return {
            "success": False,
            "error": f"Unknown model type: {model_type}"
        }
    
    # Load patient data
    patient_data_path = f"data/patient_data/{patient_id}.json"
    if not os.path.exists(patient_data_path):
        return {
            "success": False,
            "error": f"Patient data not found: {patient_id}"
        }
    
    with open(patient_data_path, "r") as f:
        patient_data = json.load(f)
    
    # Get file path
    file_path = patient_data.get("file_info", {}).get("file_path")
    if not file_path or not os.path.exists(file_path):
        return {
            "success": False,
            "error": f"Image file not found: {file_path}"
        }
    
    # Load model and run prediction
    model = MODEL_REGISTRY[model_type]
    if not model.model_loaded:
        model.load_model()
    
    try:
        # Preprocess and predict
        preprocessed = model.preprocess(file_path)
        raw_prediction = model.predict(preprocessed)
        prediction = model.postprocess(raw_prediction)
        
        # Compute priority score
        priority_score = compute_priority_score(prediction, model_type)
        
        # Update queue if requested
        if update_queue:
            # Update or add to queue
            patient_queue.add_case(
                patient_id=patient_id,
                pathology=model_type,
                score=priority_score,
                status="waiting",  # Will be updated to 'running' when processing starts
                metadata={
                    "prediction": prediction,
                    "model_type": model_type,
                    "patient_data": patient_data,
                }
            )
        
        return {
            "success": True,
            "patient_id": patient_id,
            "model_type": model_type,
            "prediction": prediction,
            "priority_score": priority_score,
            "queue_updated": update_queue
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }


@app.get("/queue")
async def get_patient_queue(limit: Optional[int] = None, status: Optional[str] = None):
    """
    Get sorted patient queue
    
    Args:
        limit: Maximum number of cases to return (None for all)
        status: Filter by status ('waiting', 'running', 'complete')
        
    Returns:
        Sorted list of patient cases by priority
    """
    queue = get_queue(status_filter=status)
    
    if limit:
        queue = queue[:limit]
    
    return {
        "queue_size": patient_queue.size(),
        "status_counts": patient_queue.get_status_counts(),
        "cases": queue
    }


@app.get("/queue/{patient_id}")
async def get_patient_queue_position(patient_id: str):
    """
    Get patient position in queue
    
    Args:
        patient_id: Patient identifier
        
    Returns:
        Patient position and details
    """
    case = patient_queue.get_case(patient_id)
    
    if case:
        # Get position in sorted queue
        queue = get_queue()
        position = None
        for idx, c in enumerate(queue):
            if c["patient_id"] == patient_id:
                position = idx + 1
                break
        
        return {
            "patient_id": patient_id,
            "position": position,
            "total_in_queue": patient_queue.size(),
            "priority_score": case["score"],
            "status": case["status"],
            "pathology": case["pathology"],
            "case_details": case
        }
    
    return {
        "patient_id": patient_id,
        "position": None,
        "message": "Patient not found in queue"
    }


@app.delete("/queue/{patient_id}")
async def remove_patient_from_queue(patient_id: str):
    """
    Remove patient from queue
    
    Args:
        patient_id: Patient identifier to remove
        
    Returns:
        Removal status
    """
    removed = remove_from_queue(patient_id)
    
    return {
        "success": removed,
        "patient_id": patient_id,
        "message": "Patient removed from queue" if removed else "Patient not found in queue"
    }


@app.patch("/queue/{patient_id}/status")
async def update_patient_status(patient_id: str, status: str = Form(...)):
    """
    Update patient case status
    
    Args:
        patient_id: Patient identifier
        status: New status ('waiting', 'running', 'complete')
        
    Returns:
        Update status
    """
    if status not in ["waiting", "running", "complete"]:
        return {
            "success": False,
            "error": f"Invalid status: {status}. Must be one of: waiting, running, complete"
        }
    
    updated = patient_queue.update_case_status(patient_id, status)
    
    if updated:
        case = patient_queue.get_case(patient_id)
        return {
            "success": True,
            "patient_id": patient_id,
            "status": status,
            "priority_score": case["score"] if case else None,
            "message": f"Status updated to {status}"
        }
    else:
        return {
            "success": False,
            "patient_id": patient_id,
            "message": "Patient not found in queue"
        }


@app.get("/queue/next")
async def get_next_patient(status: Optional[str] = "waiting"):
    """
    Get next patient case with highest priority
    
    Args:
        status: Filter by status (default: 'waiting')
        
    Returns:
        Next case or None
    """
    next_case = patient_queue.get_next_case(status_filter=status)
    
    if next_case:
        return {
            "success": True,
            "case": next_case
        }
    else:
        return {
            "success": False,
            "message": f"No cases found with status '{status}'" if status else "Queue is empty"
        }


@app.get("/get_queue_status")
async def get_queue_status(limit: Optional[int] = 10):
    """
    Get queue status with next patients and their urgency levels
    
    Args:
        limit: Maximum number of next patients to return
        
    Returns:
        Dictionary containing:
        - status_counts: Count by status
        - next_patients: List of next patients with urgency levels
        - total_in_queue: Total number of cases
    """
    # Get status counts
    status_counts = patient_queue.get_status_counts()
    
    # Get next waiting cases (sorted by priority)
    waiting_cases = get_queue(status_filter="waiting")
    next_patients = waiting_cases[:limit] if limit else waiting_cases
    
    # Format next patients with urgency levels
    formatted_next = []
    for case in next_patients:
        score = case.get("score", 0.0)
        
        # Determine urgency level based on score
        if score >= 1.5:
            urgency_level = "critical"
        elif score >= 1.0:
            urgency_level = "high"
        elif score >= 0.5:
            urgency_level = "medium"
        else:
            urgency_level = "low"
        
        formatted_next.append({
            "patient_id": case.get("patient_id"),
            "pathology": case.get("pathology"),
            "score": score,
            "urgency_level": urgency_level,
            "status": case.get("status"),
            "timestamp": case.get("timestamp"),
        })
    
    return {
        "status_counts": status_counts,
        "next_patients": formatted_next,
        "total_in_queue": patient_queue.size(),
        "waiting_count": status_counts.get("waiting", 0),
    }


@app.get("/get_result/{patient_id}")
async def get_result(patient_id: str):
    """
    Get prediction result and visualization for a patient
    
    Args:
        patient_id: Patient identifier
        
    Returns:
        Prediction result with image paths and details
    """
    case = patient_queue.get_case(patient_id)
    # If not in-memory, fallback to patient JSON file (maybe inference already saved results)
    if not case:
        patient_data_path = f"data/patient_data/{patient_id}.json"
        if os.path.exists(patient_data_path):
            with open(patient_data_path, "r") as f:
                patient_data = json.load(f)
            prediction = patient_data.get("prediction", {})
            pathology = patient_data.get("prediction", {}).get("model_type", "unknown") or "unknown"
            # craft lightweight result from saved JSON
            uncertainty = prediction.get("uncertainty", 0.0)
            uncertainty_threshold = 0.25
            needs_review = uncertainty > uncertainty_threshold
            overlay_url = None
            overlay_path = prediction.get("overlay_image_path") or prediction.get("overlay_path")
            if overlay_path and os.path.exists(overlay_path):
                overlay_url = f"/data/outputs/{os.path.basename(overlay_path)}"
            raw_url = None
            raw_path = prediction.get("raw_image_path")
            if raw_path and os.path.exists(raw_path):
                raw_url = f"/data/outputs/{os.path.basename(raw_path)}"
            # arrays
            raw_images = []
            overlays = []
            try:
                for p in prediction.get("raw_image_paths", []) or []:
                    if isinstance(p, str) and os.path.exists(p):
                        raw_images.append(f"/data/outputs/{os.path.basename(p)}")
            except Exception:
                pass
            try:
                for p in prediction.get("overlay_image_paths", []) or []:
                    if isinstance(p, str) and os.path.exists(p):
                        overlays.append(f"/data/outputs/{os.path.basename(p)}")
            except Exception:
                pass
            # compute detected from probability if explicit flags missing
            prob_for_detect = float(prediction.get("probability", prediction.get("hemorrhage_probability", 0.0)))
            detected_flag = bool(prediction.get("anomaly_detected", prediction.get("hemorrhage_detected", prob_for_detect >= 0.5)))
            
            # Determine imaging_type based on pathology
            imaging_type = "MRI" if pathology == "tumor" else "CT"
            
            result = {
                "success": True,
                "patient_id": patient_id,
                "pathology": pathology,
                "imaging_type": imaging_type,  # MRI for tumor, CT for hemorrhage
                "prediction": {
                    "probability": prediction.get("probability", prediction.get("hemorrhage_probability", 0.0)),
                    "uncertainty": uncertainty,
                    "needs_review": needs_review,
                    "uncertainty_threshold": uncertainty_threshold,
                },
                "priority_score": None,
                "status": "complete" if prediction else "pending",
                "overlay_image": overlay_url,
                "raw_image": raw_url,
                "overlays": overlays,
                "raw_images": raw_images,
                "detected": detected_flag,
            }
            return result
        return {"success": False, "error": f"Patient {patient_id} not found in queue or patient JSON"}
    
    metadata = case.get("metadata", {})
    prediction = metadata.get("prediction", {})
    pathology = case.get("pathology", "unknown")
    
    # Get uncertainty
    uncertainty = prediction.get("uncertainty", 0.0)
    uncertainty_threshold = 0.25
    needs_review = uncertainty > uncertainty_threshold
    
    # Construct image paths (assuming overlays are saved in data/outputs)
    overlay_filename = f"{pathology}_overlay_{patient_id}.png"
    overlay_path = f"data/outputs/{overlay_filename}"
    overlay_full_path = os.path.join("data", "outputs", overlay_filename)
    
    # Check if overlay exists, if not try alternative naming patterns
    overlay_url = None
    if os.path.exists(overlay_full_path):
        overlay_url = f"/data/outputs/{overlay_filename}"
    else:
        # Try to find any overlay for this patient
        outputs_dir = "data/outputs"
        if os.path.exists(outputs_dir):
            for filename in os.listdir(outputs_dir):
                if patient_id in filename and filename.endswith('.png'):
                    overlay_url = f"/data/outputs/{filename}"
                    break
    
    # Raw image URL (prefer path saved in prediction metadata)
    raw_url = None
    raw_path_meta = prediction.get("raw_image_path")
    if raw_path_meta and os.path.exists(raw_path_meta):
        raw_url = f"/data/outputs/{os.path.basename(raw_path_meta)}"
    else:
        raw_filename = f"{pathology}_raw_{patient_id}.png"
        raw_full_path = os.path.join("data", "outputs", raw_filename)
        if os.path.exists(raw_full_path):
            raw_url = f"/data/outputs/{raw_filename}"
        else:
            outputs_dir = "data/outputs"
            if os.path.exists(outputs_dir):
                for filename in os.listdir(outputs_dir):
                    if patient_id in filename and filename.startswith(f"{pathology}_raw_") and filename.endswith('.png'):
                        raw_url = f"/data/outputs/{filename}"
                        break
    # Lazy-generate raw image if still missing
    if raw_url is None:
        try:
            patient_data_path = f"data/patient_data/{patient_id}.json"
            if os.path.exists(patient_data_path):
                with open(patient_data_path, "r") as f:
                    patient_data_file = json.load(f)
                file_path = patient_data_file.get("file_info", {}).get("file_path")
                if file_path and os.path.exists(file_path):
                    model = MODEL_REGISTRY.get(pathology)
                    if model and not getattr(model, "model_loaded", False):
                        model.load_model()
                    if model:
                        preprocessed = model.preprocess(file_path)
                        if isinstance(preprocessed, np.ndarray):
                            if preprocessed.ndim == 3:
                                image_slice = preprocessed[0]
                            elif preprocessed.ndim == 4:
                                image_slice = preprocessed[0, 0]
                            else:
                                image_slice = np.asarray(preprocessed).squeeze()
                        else:
                            image_slice = np.asarray(preprocessed).squeeze()
                        from utils.explainability import save_explanation_overlay
                        raw_filename = f"{pathology}_raw_{patient_id}.png"
                        raw_full_path = save_explanation_overlay(image_slice, "data/outputs", raw_filename)
                        raw_url = f"/data/outputs/{os.path.basename(raw_full_path)}"
        except Exception as e:
            print(f"Warning: lazy raw image generation failed for {patient_id}: {e}")
    
    # compute detected from probability if explicit flags missing
    prob_for_detect = float(prediction.get("probability", prediction.get("hemorrhage_probability", 0.0)))
    detected_flag = bool(prediction.get("anomaly_detected", 
                     prediction.get("tumor_detected", 
                     prediction.get("hemorrhage_detected", prob_for_detect >= 0.5))))
    # build arrays of image URLs if available
    raw_images = []
    overlays = []
    try:
        for p in prediction.get("raw_image_paths", []) or []:
            if isinstance(p, str) and os.path.exists(p):
                raw_images.append(f"/data/outputs/{os.path.basename(p)}")
    except Exception:
        pass
    try:
        for p in prediction.get("overlay_image_paths", []) or []:
            if isinstance(p, str) and os.path.exists(p):
                overlays.append(f"/data/outputs/{os.path.basename(p)}")
    except Exception:
        pass
    # Determine imaging_type based on pathology
    imaging_type = "MRI" if pathology == "tumor" else "CT"
    
    result = {
        "success": True,
        "patient_id": patient_id,
        "pathology": pathology,
        "imaging_type": imaging_type,  # MRI for tumor, CT for hemorrhage
        "prediction": {
            "probability": prediction.get("probability", 0.0),
            "uncertainty": uncertainty,
            "needs_review": needs_review,
            "uncertainty_threshold": uncertainty_threshold,
        },
        "priority_score": case.get("score", 0.0),
        "status": case.get("status"),
        "progress": metadata.get("progress", {}),
        "overlay_image": overlay_url,
        "raw_image": raw_url,
        "overlays": overlays,
        "raw_images": raw_images,
        "detected": detected_flag,
    }
    
    # Add pathology-specific details
    if pathology == "tumor":
        result["tumor_details"] = {
            "volumes": prediction.get("volumes", {}),
            "bounding_box": prediction.get("bounding_box", {}),
        }
    elif pathology == "hemorrhage":
        result["hemorrhage_details"] = {
            "classification": prediction.get("classification", prediction.get("label", "")),
            "confidence_level": prediction.get("confidence_level", ""),
        }
    
    return result

@app.post("/predict_tumor/{patient_id}")
async def predict_tumor(
    patient_id: str,
    update_queue: bool = Form(True),
    generate_explanation: bool = Form(True),
    save_overlay: bool = Form(True)
):
    """
    Run tumor detection prediction with explanation and overlay generation

    Args:
        patient_id: Patient identifier
        update_queue: Whether to add/update in queue after prediction
        generate_explanation: Whether to generate Grad-CAM explanation
        save_overlay: Whether to save segmentation overlay image

    Returns:
        Tumor prediction results with explanation and overlay paths
    """
    model_type = "tumor"

    # Load patient data
    patient_data_path = f"data/patient_data/{patient_id}.json"
    if not os.path.exists(patient_data_path):
        return {
            "success": False,
            "error": f"Patient data not found: {patient_id}"
        }

    with open(patient_data_path, "r") as f:
        patient_data = json.load(f)

    # Get file path(s)
    file_info = patient_data.get("file_info", {})
    file_paths = file_info.get("file_paths")
    file_path = file_info.get("file_path")
    paths: List[str] = []
    if isinstance(file_paths, list) and len(file_paths) > 0:
        paths = [p for p in file_paths if isinstance(p, str)]
    elif isinstance(file_path, str) and len(file_path) > 0:
        paths = [file_path]
    else:
        return {"success": False, "error": "No image files found for patient"}
    for p in paths:
        if not os.path.exists(p):
            return {"success": False, "error": f"Image file not found: {p}"}
    
    # Load model and run prediction
    model = MODEL_REGISTRY[model_type]
    if not model.model_loaded:
        model.load_model()

    try:
        # Preprocess and predict (pass list to yield one slice per image)
        preprocessed = model.preprocess(paths)
        raw_prediction = model.predict(preprocessed)
        prediction = model.postprocess(raw_prediction)

        # Generate explanation if requested
        explanation = None
        overlay_path = None

        if generate_explanation:
            try:
                from backend.utils.explainability import explain_prediction
                explanation = explain_prediction(prediction, explanation_type='gradcam')
            except Exception as e:
                print(f"Warning: Could not generate explanation: {str(e)}")

        # Generate and save overlay if requested
        if save_overlay and prediction.get("mask") is not None:
            try:
                from backend.utils.explainability import generate_segmentation_overlay
                overlay_paths = generate_segmentation_overlay(
                    image_volume=preprocessed,
                    segmentation_mask=prediction["mask"],
                    output_dir="data/outputs",
                    prefix=f"tumor_overlay_{patient_id}",
                    alpha=0.5
                )
                if overlay_paths:
                    overlay_path = overlay_paths[0]  # Take first overlay
                    prediction["overlay_image_path"] = overlay_path
            except Exception as e:
                print(f"Warning: Could not generate overlay: {str(e)}")

        # Compute priority score
        priority_score = compute_priority_score(prediction, model_type)

        # Update queue if requested
        if update_queue:
            patient_queue.add_case(
                patient_id=patient_id,
                pathology=model_type,
                score=priority_score,
                status="waiting",
                metadata={
                    "prediction": prediction,
                    "model_type": model_type,
                    "patient_data": patient_data,
                    "explanation": explanation,
                    "overlay_path": overlay_path,
                }
            )

        return {
            "success": True,
            "patient_id": patient_id,
            "model_type": model_type,
            "prediction": prediction,
            "priority_score": priority_score,
            "explanation": explanation,
            "overlay_path": overlay_path,
            "queue_updated": update_queue
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Tumor prediction failed: {str(e)}"
        }


@app.post("/predict_hemorrhage/{patient_id}")
async def predict_hemorrhage(
    patient_id: str,
    update_queue: bool = Form(True),
    generate_explanation: bool = Form(True),
    save_overlay: bool = Form(True)
):
    """
    Run hemorrhage detection prediction with explanation and overlay generation

    Args:
        patient_id: Patient identifier
        update_queue: Whether to add/update in queue after prediction
        generate_explanation: Whether to generate Grad-CAM explanation
        save_overlay: Whether to save classification overlay image

    Returns:
        Hemorrhage prediction results with explanation and overlay paths
    """
    model_type = "hemorrhage"

    # Load patient data
    patient_data_path = f"data/patient_data/{patient_id}.json"
    if not os.path.exists(patient_data_path):
        return {
            "success": False,
            "error": f"Patient data not found: {patient_id}"
        }

    with open(patient_data_path, "r") as f:
        patient_data = json.load(f)

    # Get file path
    file_path = patient_data.get("file_info", {}).get("file_path")
    if not file_path or not os.path.exists(file_path):
        return {
            "success": False,
            "error": f"Image file not found: {file_path}"
        }

    # Load model and run prediction
    model = MODEL_REGISTRY[model_type]
    if not model.model_loaded:
        model.load_model()

    try:
        # Preprocess and predict
        preprocessed = model.preprocess(file_path)
        raw_prediction = model.predict(preprocessed)
        prediction = model.postprocess(raw_prediction)

        # Generate explanation if requested
        explanation = None
        overlay_path = None

        if generate_explanation:
            try:
                from backend.utils.explainability import explain_prediction
                explanation = explain_prediction(prediction, explanation_type='gradcam')
            except Exception as e:
                print(f"Warning: Could not generate explanation: {str(e)}")

        # Generate and save overlay if requested (for classification)
        if save_overlay:
            try:
                from backend.utils.explainability import overlay_heatmap, save_explanation_overlay
                import cv2

                # Get original image slice for overlay
                if preprocessed.ndim == 3:
                    # Take middle slice for 3D volume
                    slice_idx = preprocessed.shape[0] // 2
                    image_slice = preprocessed[slice_idx]
                else:
                    image_slice = preprocessed.squeeze()

                # Always save raw processed slice image for side-by-side view
                raw_filename = f"{model_type}_raw_{patient_id}.png"
                raw_path = save_explanation_overlay(image_slice, "data/outputs", raw_filename)
                prediction["raw_image_path"] = raw_path

                # If we have a proper heatmap from explanation, use it, else save raw slice per user request
                have_heatmap = False
                if explanation and "visualization" not in explanation:
                    hm = explanation.get("feature_importance", {}).get("prediction_probability", None)
                    if isinstance(hm, np.ndarray):
                        heatmap_array = hm.astype(np.float32)
                        have_heatmap = True
                if have_heatmap:
                    overlay = overlay_heatmap(image_slice, heatmap_array, alpha=0.4)
                else:
                    overlay = image_slice

                filename = f"hemorrhage_overlay_{patient_id}.png"
                overlay_path = save_explanation_overlay(overlay, "data/outputs", filename)
                prediction["overlay_image_path"] = overlay_path

            except Exception as e:
                print(f"Warning: Could not generate overlay: {str(e)}")

        # Compute priority score
        priority_score = compute_priority_score(prediction, model_type)

        # Update queue if requested
        if update_queue:
            patient_queue.add_case(
                patient_id=patient_id,
                pathology=model_type,
                score=priority_score,
                status="waiting",
                metadata={
                    "prediction": prediction,
                    "model_type": model_type,
                    "patient_data": patient_data,
                    "explanation": explanation,
                    "overlay_path": overlay_path,
                }
            )

        return {
            "success": True,
            "patient_id": patient_id,
            "model_type": model_type,
            "prediction": prediction,
            "priority_score": priority_score,
            "explanation": explanation,
            "overlay_path": overlay_path,
            "queue_updated": update_queue
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Hemorrhage prediction failed: {str(e)}"
        }

