// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:neurotriage_app/main.dart';

void main() {
  testWidgets('NeuroTriage app smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const NeuroTriageApp());

    // Verify that the app builds and shows the PatientFormPage
    expect(find.text('NeuroTriage System'), findsOneWidget);
    expect(find.text('Brain Anomaly Detection'), findsOneWidget);
    expect(find.text('Patient Intake Form'), findsOneWidget);

    // Verify key UI elements are present
    expect(find.text('Full Name'), findsOneWidget);
    expect(find.text('Age'), findsOneWidget);
    expect(find.text('Gender'), findsOneWidget);
    expect(find.text('Imaging Type'), findsOneWidget);
    expect(find.text('Symptoms (select all that apply):'), findsOneWidget);
    expect(find.text('Select File'), findsOneWidget);
    expect(find.text('Submit Patient Data'), findsOneWidget);
  });
}
