import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart';

class ApiService {
  // Backend URL - automatically configured based on platform
  static String get baseUrl {
    if (kIsWeb) {
      // For web, prefer 127.0.0.1 to avoid some browsers blocking localhost in fetch
      return 'http://127.0.0.1:8000';
    } else if (Platform.isAndroid) {
      // For Android emulator
      return 'http://10.0.2.2:8000';
    } else if (Platform.isIOS) {
      // For iOS simulator
      return 'http://localhost:8000';
    } else {
      // For desktop or other platforms in development, default to localhost
      return 'http://localhost:8000';
    }
  }

  static Future<Map<String, dynamic>> uploadPatientData({
    required String name,
    required int age,
    required String gender,
    required String imagingType,
    required List<String> symptoms,
    String? history,
    String? notes,
    // Single-file legacy
    File? file,
    Uint8List? fileBytes,
    String? fileName,
    // Multi-file new API (preferred)
    List<File>? files,
    List<Uint8List>? fileBytesList,
    List<String>? fileNames,
    String? scanType,
    // Brain and Bone folder uploads (CT)
    List<File>? brainFiles,
    List<Uint8List>? brainFileBytesList,
    List<String>? brainFileNames,
    List<File>? boneFiles,
    List<Uint8List>? boneFileBytesList,
    List<String>? boneFileNames,
    // MRI folder uploads
    List<File>? mriFiles,
    List<Uint8List>? mriFileBytesList,
    List<String>? mriFileNames,
  }) async {
    try {
      // Prepare JSON patient data
      final patientDataJson = jsonEncode({
        'demographics': {
          'name': name,
          'age': age,
          'gender': gender,
        },
        'clinical_context': {
          'imaging_type': imagingType,
          'symptoms': symptoms,
          'history': history,
          'notes': notes,
        },
        'scan_type': scanType ?? 'CT',
      });

      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/upload_patient_data'),
      );

      // Add JSON patient data
      request.fields['patient_data_json'] = patientDataJson;

      // Check for brain/bone folder mode first (highest priority)
      final hasBrainBone = (brainFiles != null && brainFiles.isNotEmpty) || 
                           (brainFileBytesList != null && brainFileBytesList.isNotEmpty);
      
      if (hasBrainBone) {
        // Upload brain files
        final brainNames = brainFileNames ?? [];
        int brainCount = brainFiles?.length ?? brainFileBytesList!.length;
        for (int i = 0; i < brainCount; i++) {
          final filename = (i < brainNames.length && brainNames[i].isNotEmpty) ? brainNames[i] : 'brain_${i + 1}.jpg';
          if (brainFiles != null && i < brainFiles.length) {
            final f = brainFiles[i];
            final stream = f.openRead();
            final length = await f.length();
            request.files.add(http.MultipartFile('brain_files', stream, length, filename: filename));
          } else if (brainFileBytesList != null && i < brainFileBytesList.length) {
            final bytes = brainFileBytesList[i];
            request.files.add(http.MultipartFile.fromBytes('brain_files', bytes, filename: filename));
          }
        }
        
        // Upload bone files
        final boneNames = boneFileNames ?? [];
        int boneCount = boneFiles?.length ?? boneFileBytesList!.length;
        for (int i = 0; i < boneCount; i++) {
          final filename = (i < boneNames.length && boneNames[i].isNotEmpty) ? boneNames[i] : 'bone_${i + 1}.jpg';
          if (boneFiles != null && i < boneFiles.length) {
            final f = boneFiles[i];
            final stream = f.openRead();
            final length = await f.length();
            request.files.add(http.MultipartFile('bone_files', stream, length, filename: filename));
          } else if (boneFileBytesList != null && i < boneFileBytesList.length) {
            final bytes = boneFileBytesList[i];
            request.files.add(http.MultipartFile.fromBytes('bone_files', bytes, filename: filename));
          }
        }
      } else if ((mriFiles != null && mriFiles.isNotEmpty) || (mriFileBytesList != null && mriFileBytesList.isNotEmpty)) {
        // MRI folder mode
        final mriNames = mriFileNames ?? [];
        int mriCount = mriFiles?.length ?? mriFileBytesList!.length;
        for (int i = 0; i < mriCount; i++) {
          final filename = (i < mriNames.length && mriNames[i].isNotEmpty) ? mriNames[i] : 'mri_${i + 1}.jpg';
          if (mriFiles != null && i < mriFiles.length) {
            final f = mriFiles[i];
            final stream = f.openRead();
            final length = await f.length();
            request.files.add(http.MultipartFile('mri_files', stream, length, filename: filename));
          } else if (mriFileBytesList != null && i < mriFileBytesList.length) {
            final bytes = mriFileBytesList[i];
            request.files.add(http.MultipartFile.fromBytes('mri_files', bytes, filename: filename));
          }
        }
      } else {
        // Fallback to multi-file payload if provided
        final hasMultiFiles = (files != null && files.isNotEmpty) || (fileBytesList != null && fileBytesList.isNotEmpty);
        if (hasMultiFiles) {
        // Validate names length
        final names = fileNames ?? [];
        int count = files?.length ?? fileBytesList!.length;
        for (int i = 0; i < count; i++) {
          final filename = (i < names.length && names[i].isNotEmpty) ? names[i] : 'image_${i + 1}.dat';
          if (files != null && i < files.length) {
            final f = files[i];
            final stream = f.openRead();
            final length = await f.length();
            request.files.add(http.MultipartFile('files', stream, length, filename: filename));
          } else if (fileBytesList != null && i < fileBytesList.length) {
            final bytes = fileBytesList[i];
            request.files.add(http.MultipartFile.fromBytes('files', bytes, filename: filename));
          }
        }
      } else {
        // Legacy single-file
        if (file != null) {
          var fileStream = file.openRead();
          var fileLength = await file.length();
          var multipartFile = http.MultipartFile(
            'file',
            fileStream,
            fileLength,
            filename: (fileName ?? 'upload.dat'),
          );
          request.files.add(multipartFile);
        } else if (fileBytes != null) {
          var multipartFile = http.MultipartFile.fromBytes(
            'file',
            fileBytes,
            filename: (fileName ?? 'upload.dat'),
          );
          request.files.add(multipartFile);
        }
      }
      }

      // Send request
      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      } else {
        return {
          'success': false,
          'error': 'Server error: ${response.statusCode}',
        };
      }
    } catch (e) {
      return {
        'success': false,
        'error': 'Network error: $e',
      };
    }
  }

  static Future<Map<String, dynamic>> checkHealth() async {
    try {
      var response = await http.get(Uri.parse('$baseUrl/health'));
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      }
      return {'status': 'unhealthy'};
    } catch (e) {
      return {'status': 'error', 'error': e.toString()};
    }
  }

  static Future<Map<String, dynamic>> getResult(
    String patientId, {
    bool waitForCompletion = true,
    int timeoutSeconds = 180,
    int pollIntervalSeconds = 2,
  }) async {
    final uri = Uri.parse('$baseUrl/get_result/$patientId');
    final endTime = DateTime.now().add(Duration(seconds: timeoutSeconds));

    try {
      while (true) {
        var response = await http.get(uri);

        if (response.statusCode == 200) {
          final body = jsonDecode(response.body) as Map<String, dynamic>;

          // If server reports success=true and not pending, return immediately
          final success = body['success'] == true;
          final status = (body['status'] ?? '').toString().toLowerCase();

          final isPending = !success ||
              status.contains('pending') ||
              status.contains('processing') ||
              (body['prediction'] == null);

          if (!isPending) {
            return body;
          }

          // Still processing: either poll again or return current payload
          if (waitForCompletion && DateTime.now().isBefore(endTime)) {
            await Future.delayed(Duration(seconds: pollIntervalSeconds));
            continue;
          }

          // Timeout or not waiting: return current payload (may contain placeholders)
          return body;
        } else if (response.statusCode == 202) {
          // Accepted -> processing
          if (waitForCompletion && DateTime.now().isBefore(endTime)) {
            await Future.delayed(Duration(seconds: pollIntervalSeconds));
            continue;
          }
          return {
            'success': false,
            'error': 'Processing (202)',
            'statusCode': 202,
          };
        } else {
          return {
            'success': false,
            'error': 'Server error: ${response.statusCode}',
          };
        }
      }
    } catch (e) {
      return {
        'success': false,
        'error': 'Network error: $e',
      };
    }
  }

  static Future<Map<String, dynamic>> getQueueStatus({int? limit}) async {
    try {
      var uri = Uri.parse('$baseUrl/get_queue_status');
      if (limit != null) {
        uri = uri.replace(queryParameters: {'limit': limit.toString()});
      }

      var response = await http.get(uri);

      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      } else {
        return {
          'success': false,
          'error': 'Server error: ${response.statusCode}',
        };
      }
    } catch (e) {
      return {
        'success': false,
        'error': 'Network error: $e',
      };
    }
  }

  static Future<Map<String, dynamic>> getOverlayImage(String imagePath) async {
    try {
      // Ensure we build a valid URL even if backend returns a path starting with '/'
      final path = imagePath.startsWith('/') ? imagePath : '/$imagePath';
      var response = await http.get(Uri.parse('$baseUrl$path'));

      if (response.statusCode == 200) {
        return {
          'success': true,
          'imageBytes': response.bodyBytes,
        };
      } else {
        return {
          'success': false,
          'error': 'Failed to load image: ${response.statusCode}',
        };
      }
    } catch (e) {
      return {
        'success': false,
        'error': 'Network error: $e',
      };
    }
  }
}

