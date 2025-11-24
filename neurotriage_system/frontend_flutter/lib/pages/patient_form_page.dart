import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:io';
import 'dart:typed_data';
import '../services/api_service.dart';
import 'results_page.dart';

class PatientFormPage extends StatefulWidget {
  const PatientFormPage({super.key});

  @override
  State<PatientFormPage> createState() => _PatientFormPageState();
}

class _PatientFormPageState extends State<PatientFormPage> {
  final _formKey = GlobalKey<FormState>();
  final _nameController = TextEditingController();
  final _ageController = TextEditingController();
  final _historyController = TextEditingController();
  final _notesController = TextEditingController();

  String? _selectedGender;
  String? _selectedImagingType;
  final Set<String> _selectedSymptoms = {};
  File? _selectedFile;
  Uint8List? _selectedFileBytes;
  String? _selectedFileName;
  List<File> _selectedFiles = [];
  List<Uint8List> _selectedFileBytesList = [];
  List<String> _selectedFileNames = [];
  
  // Brain and Bone folder files
  List<File> _brainFiles = [];
  List<Uint8List> _brainFileBytesList = [];
  List<String> _brainFileNames = [];
  List<File> _boneFiles = [];
  List<Uint8List> _boneFileBytesList = [];
  List<String> _boneFileNames = [];
  
  // MRI folder files
  List<File> _mriFiles = [];
  List<Uint8List> _mriFileBytesList = [];
  List<String> _mriFileNames = [];
  
  // Scan type selection
  String _scanType = 'CT'; // 'CT' or 'MRI'
  
  bool _isUploading = false;

  final List<String> _symptoms = [
    'Headache',
    'Nausea/Vomiting',
    'Loss of consciousness',
    'Seizures',
    'Speech difficulties',
    'Motor/Balance issues',
    'Vision problems',
    'Cognitive changes',
    'Memory problems',
    'Weakness/Numbness',
    'Trauma history',
    'Focal neurological deficits',
  ];

  @override
  void dispose() {
    _nameController.dispose();
    _ageController.dispose();
    _historyController.dispose();
    _notesController.dispose();
    super.dispose();
  }

  Future<void> _pickFile() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        allowMultiple: true,
        type: FileType.custom,
        allowedExtensions: ['dcm', 'dicom', 'nii', 'gz', 'jpg', 'jpeg', 'png'],
      );

      if (result != null) {
        setState(() {
          _selectedFiles = [];
          _selectedFileBytesList = [];
          _selectedFileNames = [];
          for (final file in result.files) {
            _selectedFileNames.add(file.name);
            if (file.bytes != null) {
              _selectedFileBytesList.add(file.bytes!);
            } else if (file.path != null) {
              _selectedFiles.add(File(file.path!));
            }
          }
          if (_selectedFileNames.isNotEmpty) {
            _selectedFileName = _selectedFileNames.first;
          }
          if (_selectedFiles.isNotEmpty) {
            _selectedFile = _selectedFiles.first;
            _selectedFileBytes = null;
          } else if (_selectedFileBytesList.isNotEmpty) {
            _selectedFileBytes = _selectedFileBytesList.first;
            _selectedFile = null;
          }
        });
      }
    } catch (e) {
      _showSnackBar('Error picking file: $e', isError: true);
    }
  }

  Future<void> _pickBrainFolder() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        allowMultiple: true,
        type: FileType.custom,
        allowedExtensions: ['jpg', 'jpeg', 'png'],
        dialogTitle: 'Select Brain Window Images',
      );

      if (result != null) {
        setState(() {
          _brainFiles = [];
          _brainFileBytesList = [];
          _brainFileNames = [];
          for (final file in result.files) {
            _brainFileNames.add(file.name);
            if (file.bytes != null) {
              _brainFileBytesList.add(file.bytes!);
            } else if (file.path != null) {
              _brainFiles.add(File(file.path!));
            }
          }
        });
      }
    } catch (e) {
      _showSnackBar('Error picking brain files: $e', isError: true);
    }
  }

  Future<void> _pickBoneFolder() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        allowMultiple: true,
        type: FileType.custom,
        allowedExtensions: ['jpg', 'jpeg', 'png'],
        dialogTitle: 'Select Bone Window Images',
      );

      if (result != null) {
        setState(() {
          _boneFiles = [];
          _boneFileBytesList = [];
          _boneFileNames = [];
          for (final file in result.files) {
            _boneFileNames.add(file.name);
            if (file.bytes != null) {
              _boneFileBytesList.add(file.bytes!);
            } else if (file.path != null) {
              _boneFiles.add(File(file.path!));
            }
          }
        });
      }
    } catch (e) {
      _showSnackBar('Error picking bone files: $e', isError: true);
    }
  }

  Future<void> _pickMRIFolder() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        allowMultiple: true,
        type: FileType.custom,
        allowedExtensions: ['jpg', 'jpeg', 'png', 'nii', 'gz'],
        dialogTitle: 'Select MRI Scan Images',
      );

      if (result != null) {
        setState(() {
          _mriFiles = [];
          _mriFileBytesList = [];
          _mriFileNames = [];
          for (final file in result.files) {
            _mriFileNames.add(file.name);
            if (file.bytes != null) {
              _mriFileBytesList.add(file.bytes!);
            } else if (file.path != null) {
              _mriFiles.add(File(file.path!));
            }
          }
        });
      }
    } catch (e) {
      _showSnackBar('Error picking MRI files: $e', isError: true);
    }
  }

  void _showSnackBar(String message, {bool isError = false}) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            Icon(
              isError ? Icons.error_outline : Icons.check_circle_outline,
              color: Colors.white,
            ),
            const SizedBox(width: 12),
            Expanded(child: Text(message)),
          ],
        ),
        backgroundColor: isError ? MedicalColors.danger : MedicalColors.success,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        duration: Duration(seconds: isError ? 4 : 2),
      ),
    );
  }

  Future<void> _submitForm() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    // Check scan type specific requirements
    if (_scanType == 'CT') {
      // CT requires brain + bone folders
      final hasBrainBoneMode = _brainFileNames.isNotEmpty && _boneFileNames.isNotEmpty;
      
      if (!hasBrainBoneMode) {
        _showSnackBar('Please select both brain and bone folder images for CT scan', isError: true);
        return;
      }
      
      if (_brainFileNames.length != _boneFileNames.length) {
        _showSnackBar('Brain and bone folders must have the same number of slices', isError: true);
        return;
      }
    } else if (_scanType == 'MRI') {
      // MRI requires single folder
      final hasMRIFiles = _mriFileNames.isNotEmpty;
      
      if (!hasMRIFiles) {
        _showSnackBar('Please select MRI scan images', isError: true);
        return;
      }
    }

    if (_selectedImagingType == null) {
      _showSnackBar('Please select imaging type', isError: true);
      return;
    }

    setState(() => _isUploading = true);

    try {
      final result = await ApiService.uploadPatientData(
        name: _nameController.text,
        age: int.parse(_ageController.text),
        gender: _selectedGender!,
        imagingType: _selectedImagingType!,
        symptoms: _selectedSymptoms.toList(),
        history: _historyController.text.isEmpty ? null : _historyController.text,
        notes: _notesController.text.isEmpty ? null : _notesController.text,
        scanType: _scanType,
        file: _selectedFiles.isEmpty ? _selectedFile : null,
        fileBytes: _selectedFileBytesList.isEmpty ? _selectedFileBytes : null,
        fileName: _selectedFileName,
        files: _selectedFiles.isNotEmpty ? _selectedFiles : null,
        fileBytesList: _selectedFileBytesList.isNotEmpty ? _selectedFileBytesList : null,
        fileNames: _selectedFileNames.isNotEmpty ? _selectedFileNames : null,
        // CT scan parameters (brain/bone folders)
        brainFiles: _scanType == 'CT' && _brainFiles.isNotEmpty ? _brainFiles : null,
        brainFileBytesList: _scanType == 'CT' && _brainFileBytesList.isNotEmpty ? _brainFileBytesList : null,
        brainFileNames: _scanType == 'CT' && _brainFileNames.isNotEmpty ? _brainFileNames : null,
        boneFiles: _scanType == 'CT' && _boneFiles.isNotEmpty ? _boneFiles : null,
        boneFileBytesList: _scanType == 'CT' && _boneFileBytesList.isNotEmpty ? _boneFileBytesList : null,
        boneFileNames: _scanType == 'CT' && _boneFileNames.isNotEmpty ? _boneFileNames : null,
        // MRI scan parameters (single folder)
        mriFiles: _scanType == 'MRI' && _mriFiles.isNotEmpty ? _mriFiles : null,
        mriFileBytesList: _scanType == 'MRI' && _mriFileBytesList.isNotEmpty ? _mriFileBytesList : null,
        mriFileNames: _scanType == 'MRI' && _mriFileNames.isNotEmpty ? _mriFileNames : null,
      );

      if (!mounted) return;

      if (result['success'] == true) {
        final patientId = result['patient_id'];
        
        _showSnackBar('Patient data uploaded successfully!');
        
        await Future.delayed(const Duration(milliseconds: 500));
        
        if (mounted) {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => ResultsPage(patientId: patientId),
            ),
          );
        }
        
        _resetForm();
      } else {
        _showSnackBar('Error: ${result['error'] ?? 'Unknown error'}', isError: true);
      }
    } catch (e) {
      _showSnackBar('Error uploading data: $e', isError: true);
    } finally {
      if (mounted) setState(() => _isUploading = false);
    }
  }

  void _resetForm() {
    if (!mounted) return;
    _formKey.currentState!.reset();
    _nameController.clear();
    _ageController.clear();
    _historyController.clear();
    _notesController.clear();
    setState(() {
      _selectedGender = null;
      _selectedImagingType = null;
      _selectedSymptoms.clear();
      _selectedFile = null;
      _selectedFileBytes = null;
      _selectedFileName = null;
      _selectedFiles = [];
      _selectedFileBytesList = [];
      _selectedFileNames = [];
      _brainFiles = [];
      _brainFileBytesList = [];
      _brainFileNames = [];
      _boneFiles = [];
      _boneFileBytesList = [];
      _boneFileNames = [];
      _mriFiles = [];
      _mriFileBytesList = [];
      _mriFileNames = [];
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: MedicalColors.lightGray,
      appBar: AppBar(
        elevation: 0,
        title: const Text(
          'Patient Intake',
          style: TextStyle(
            fontWeight: FontWeight.w600,
            color: Colors.white,
          ),
        ),
        backgroundColor: MedicalColors.primary,
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            // Hero Header
            Container(
              width: double.infinity,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [MedicalColors.primary, MedicalColors.accent],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
              ),
              padding: const EdgeInsets.all(32),
              child: Column(
                children: [
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.2),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(
                      Icons.medical_information,
                      size: 64,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    'NeuroTriage System',
                    style: TextStyle(
                      fontSize: 28,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'AI-Powered Brain Anomaly Detection',
                    style: TextStyle(
                      fontSize: 16,
                      color: Colors.white.withOpacity(0.9),
                    ),
                  ),
                ],
              ),
            ),

            // Form Content
            Padding(
              padding: const EdgeInsets.all(20),
              child: Form(
                key: _formKey,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _buildDemographicsSection(),
                    const SizedBox(height: 20),
                    _buildClinicalSection(),
                    const SizedBox(height: 20),
                    _buildScanTypeSelector(),
                    const SizedBox(height: 20),
                    _buildFileUploadSection(),
                    const SizedBox(height: 20),
                    _buildNotesSection(),
                    const SizedBox(height: 32),
                    _buildSubmitButton(),
                    const SizedBox(height: 40),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDemographicsSection() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: MedicalColors.accent.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: const Icon(
                  Icons.person_outline,
                  color: MedicalColors.accent,
                  size: 24,
                ),
              ),
              const SizedBox(width: 12),
              const Text(
                'Patient Demographics',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: MedicalColors.primary,
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),
          TextFormField(
            controller: _nameController,
            decoration: InputDecoration(
              labelText: 'Full Name *',
              hintText: 'Enter patient full name',
              prefixIcon: const Icon(Icons.person, color: MedicalColors.accent),
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
              filled: true,
              fillColor: MedicalColors.lightGray,
            ),
            validator: (value) => 
                value == null || value.trim().isEmpty ? 'Please enter patient name' : null,
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: TextFormField(
                  controller: _ageController,
                  decoration: InputDecoration(
                    labelText: 'Age *',
                    hintText: 'Years',
                    prefixIcon: const Icon(Icons.cake, color: MedicalColors.accent),
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                    filled: true,
                    fillColor: MedicalColors.lightGray,
                  ),
                  keyboardType: TextInputType.number,
                  validator: (value) {
                    if (value == null || value.trim().isEmpty) return 'Required';
                    final age = int.tryParse(value);
                    if (age == null || age < 0 || age > 150) return 'Invalid age';
                    return null;
                  },
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: DropdownButtonFormField<String>(
                  value: _selectedGender,
                  decoration: InputDecoration(
                    labelText: 'Gender *',
                    prefixIcon: const Icon(Icons.wc, color: MedicalColors.accent),
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                    filled: true,
                    fillColor: MedicalColors.lightGray,
                  ),
                  items: ['Male', 'Female', 'Other']
                      .map((g) => DropdownMenuItem(value: g, child: Text(g)))
                      .toList(),
                  onChanged: (value) => setState(() => _selectedGender = value),
                  validator: (value) => value == null ? 'Required' : null,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildClinicalSection() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: MedicalColors.danger.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: const Icon(
                  Icons.medical_services,
                  color: MedicalColors.danger,
                  size: 24,
                ),
              ),
              const SizedBox(width: 12),
              const Text(
                'Clinical Information',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: MedicalColors.primary,
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),
          DropdownButtonFormField<String>(
            value: _selectedImagingType,
            decoration: InputDecoration(
              labelText: 'Imaging Modality *',
              prefixIcon: const Icon(Icons.scanner, color: MedicalColors.accent),
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
              filled: true,
              fillColor: MedicalColors.lightGray,
            ),
            items: ['MRI', 'CT']
                .map((type) => DropdownMenuItem(value: type, child: Text(type)))
                .toList(),
            onChanged: (value) => setState(() => _selectedImagingType = value),
            validator: (value) => value == null ? 'Please select imaging type' : null,
          ),
          const SizedBox(height: 20),
          const Text(
            'Presenting Symptoms:',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w600,
              color: MedicalColors.primary,
            ),
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: _symptoms.map((symptom) {
              final isSelected = _selectedSymptoms.contains(symptom);
              return ChoiceChip(
                label: Text(symptom),
                selected: isSelected,
                onSelected: (selected) {
                  setState(() {
                    if (selected) {
                      _selectedSymptoms.add(symptom);
                    } else {
                      _selectedSymptoms.remove(symptom);
                    }
                  });
                },
                selectedColor: MedicalColors.accent.withOpacity(0.2),
                labelStyle: TextStyle(
                  color: isSelected ? MedicalColors.accent : Colors.grey.shade700,
                  fontWeight: isSelected ? FontWeight.w600 : FontWeight.normal,
                ),
                side: BorderSide(
                  color: isSelected ? MedicalColors.accent : Colors.grey.shade300,
                ),
              );
            }).toList(),
          ),
          const SizedBox(height: 20),
          TextFormField(
            controller: _historyController,
            decoration: InputDecoration(
              labelText: 'Medical History',
              hintText: 'Enter relevant medical history...',
              prefixIcon: const Icon(Icons.history, color: MedicalColors.accent),
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
              filled: true,
              fillColor: MedicalColors.lightGray,
            ),
            maxLines: 3,
          ),
        ],
      ),
    );
  }

  Widget _buildScanTypeSelector() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Scan Type',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: MedicalColors.primary,
            ),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: _buildScanTypeButton('CT', Icons.local_hospital, 'Brain + Bone Windows'),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: _buildScanTypeButton('MRI', Icons.sick, 'T1/T2/FLAIR Images'),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildScanTypeButton(String type, IconData icon, String subtitle) {
    final isSelected = _scanType == type;
    return GestureDetector(
      onTap: () => setState(() {
        _scanType = type;
        // Clear opposite type files
        if (type == 'CT') {
          _mriFiles = [];
          _mriFileBytesList = [];
          _mriFileNames = [];
        } else {
          _brainFiles = [];
          _brainFileBytesList = [];
          _brainFileNames = [];
          _boneFiles = [];
          _boneFileBytesList = [];
          _boneFileNames = [];
        }
      }),
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          border: Border.all(
            color: isSelected ? MedicalColors.accent : Colors.grey.shade300,
            width: 2,
          ),
          borderRadius: BorderRadius.circular(12),
          color: isSelected ? MedicalColors.accent.withOpacity(0.1) : Colors.white,
        ),
        child: Column(
          children: [
            Icon(
              icon,
              color: isSelected ? MedicalColors.accent : Colors.grey,
              size: 36,
            ),
            const SizedBox(height: 8),
            Text(
              type,
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: isSelected ? MedicalColors.accent : Colors.grey,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              subtitle,
              style: TextStyle(
                fontSize: 11,
                color: Colors.grey.shade600,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildFileUploadSection() {
    final hasBrain = _brainFileNames.isNotEmpty;
    final hasBone = _boneFileNames.isNotEmpty;
    final hasMRI = _mriFileNames.isNotEmpty;

    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: MedicalColors.success.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: const Icon(
                  Icons.upload_file,
                  color: MedicalColors.success,
                  size: 24,
                ),
              ),
              const SizedBox(width: 12),
              Text(
                _scanType == 'CT' ? 'CT Imaging Files' : 'MRI Imaging Files',
                style: const TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: MedicalColors.primary,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Text(
            _scanType == 'CT'
                ? 'Upload brain and bone windowed slices (JPEG/PNG)'
                : 'Upload MRI scan images (JPEG/PNG/NIfTI)',
            style: TextStyle(
              fontSize: 13,
              color: Colors.grey.shade600,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 20),
          
          // Dynamic content based on scan type
          if (_scanType == 'CT') ...[
          
          // Brain folder upload
          GestureDetector(
            onTap: _pickBrainFolder,
            child: Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                border: Border.all(
                  color: hasBrain ? MedicalColors.success : MedicalColors.accent,
                  width: 2,
                ),
                borderRadius: BorderRadius.circular(12),
                color: hasBrain 
                    ? MedicalColors.success.withOpacity(0.1)
                    : MedicalColors.accent.withOpacity(0.05),
              ),
              child: Row(
                children: [
                  Icon(
                    hasBrain ? Icons.check_circle : Icons.folder_outlined,
                    color: hasBrain ? MedicalColors.success : MedicalColors.accent,
                    size: 32,
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          hasBrain 
                              ? '${_brainFileNames.length} Brain Window Slices' 
                              : 'Click to Select Brain Window Folder',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: hasBrain ? MedicalColors.success : MedicalColors.accent,
                          ),
                        ),
                        if (!hasBrain)
                          Text(
                            'Select all brain window images',
                            style: TextStyle(
                              fontSize: 13,
                              color: Colors.grey.shade600,
                            ),
                          ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
          
          const SizedBox(height: 16),
          
          // Bone folder upload
          GestureDetector(
            onTap: _pickBoneFolder,
            child: Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                border: Border.all(
                  color: hasBone ? MedicalColors.success : MedicalColors.accent,
                  width: 2,
                ),
                borderRadius: BorderRadius.circular(12),
                color: hasBone 
                    ? MedicalColors.success.withOpacity(0.1)
                    : MedicalColors.accent.withOpacity(0.05),
              ),
              child: Row(
                children: [
                  Icon(
                    hasBone ? Icons.check_circle : Icons.folder_outlined,
                    color: hasBone ? MedicalColors.success : MedicalColors.accent,
                    size: 32,
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          hasBone 
                              ? '${_boneFileNames.length} Bone Window Slices' 
                              : 'Click to Select Bone Window Folder',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: hasBone ? MedicalColors.success : MedicalColors.accent,
                          ),
                        ),
                        if (!hasBone)
                          Text(
                            'Select all bone window images',
                            style: TextStyle(
                              fontSize: 13,
                              color: Colors.grey.shade600,
                            ),
                          ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
          
          if (hasBrain && hasBone && _brainFileNames.length != _boneFileNames.length)
            Padding(
              padding: const EdgeInsets.only(top: 12),
              child: Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: MedicalColors.danger.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: MedicalColors.danger),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.warning, color: MedicalColors.danger),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'Slice count mismatch! Brain: ${_brainFileNames.length}, Bone: ${_boneFileNames.length}',
                        style: const TextStyle(
                          color: MedicalColors.danger,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ] else ...[
            // MRI folder upload
            GestureDetector(
              onTap: _pickMRIFolder,
              child: Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  border: Border.all(
                    color: hasMRI ? MedicalColors.success : MedicalColors.accent,
                    width: 2,
                  ),
                  borderRadius: BorderRadius.circular(12),
                  color: hasMRI
                      ? MedicalColors.success.withOpacity(0.1)
                      : MedicalColors.accent.withOpacity(0.05),
                ),
                child: Row(
                  children: [
                    Icon(
                      hasMRI ? Icons.check_circle : Icons.folder_outlined,
                      color: hasMRI ? MedicalColors.success : MedicalColors.accent,
                      size: 32,
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            hasMRI
                                ? '${_mriFileNames.length} MRI Images Selected'
                                : 'Click to Select MRI Images',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                              color: hasMRI ? MedicalColors.success : MedicalColors.accent,
                            ),
                          ),
                          if (!hasMRI)
                            Text(
                              'Select all MRI scan images',
                              style: TextStyle(
                                fontSize: 13,
                                color: Colors.grey.shade600,
                              ),
                            ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildNotesSection() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: MedicalColors.info.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: const Icon(
                  Icons.note_alt_outlined,
                  color: MedicalColors.info,
                  size: 24,
                ),
              ),
              const SizedBox(width: 12),
              const Text(
                'Additional Notes',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: MedicalColors.primary,
                ),
              ),
              const SizedBox(width: 8),
              Text(
                '(Optional)',
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.grey.shade500,
                  fontStyle: FontStyle.italic,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          TextFormField(
            controller: _notesController,
            decoration: InputDecoration(
              hintText: 'Enter any additional clinical notes or observations...',
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
              filled: true,
              fillColor: MedicalColors.lightGray,
            ),
            maxLines: 4,
          ),
        ],
      ),
    );
  }

  Widget _buildSubmitButton() {
    return SizedBox(
      width: double.infinity,
      height: 56,
      child: ElevatedButton(
        onPressed: _isUploading ? null : _submitForm,
        style: ElevatedButton.styleFrom(
          backgroundColor: MedicalColors.accent,
          disabledBackgroundColor: Colors.grey.shade300,
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
        child: _isUploading
            ? const SizedBox(
                height: 24,
                width: 24,
                child: CircularProgressIndicator(
                  strokeWidth: 3,
                  valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                ),
              )
            : const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.send, color: Colors.white),
                  SizedBox(width: 12),
                  Text(
                    'Submit for Analysis',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                ],
              ),
      ),
    );
  }
}
