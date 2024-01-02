import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import 'image_utils.dart';

class ScannerScreen extends StatefulWidget {
  @override
  _ScannerScreenState createState() => _ScannerScreenState();
}

class _ScannerScreenState extends State<ScannerScreen> {
  bool initialized = false;
  File? imageFile;
  @override
  void initState() {
    super.initState();
    initialize();
  }

  Future<void> initialize() async {}

  Future<void> pickImage() async {
    ImagePicker().pickImage(source: ImageSource.gallery).then((value) {
      imageFile = value?.path == null ? null : File(value!.path);
      setState(() {});
    });
  }

  ///DONT REMOVE THIS CODE: The is for CNN
  Future<String> predict(File imageFile) async {
    List<String> classNames = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'];
    List<List<List<List<double>>>> input = preprocessInput(imageFile, 256);
    List<List<double>> output = List.generate(1, (index) => List.filled(3, 0.0));
    Interpreter interpreter = await Interpreter.fromAsset('model.tflite');

    interpreter.run(input, output);
    int predictedClass = output[0].indexOf(output[0].reduce((a, b) => a > b ? a : b));
    String predictedClassName = classNames[predictedClass];

    return predictedClassName;
  }

  Future<void> predictUsingSVM() async {}

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Flutter Camera Demo'),
      ),
      body: Center(
        child: imageFile == null
            ? ElevatedButton(
                onPressed: pickImage,
                child: Text('Choose image'),
              )
            : Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  SizedBox(
                    height: MediaQuery.of(context).size.width * 0.7,
                    width: MediaQuery.of(context).size.width * 0.7,
                    child: Image.file(
                      imageFile!,
                      fit: BoxFit.cover,
                    ),
                  ),
                  const SizedBox(
                    height: 10,
                  ),
                  ElevatedButton(
                      onPressed: () async {
                        print(await predict(imageFile!));
                      },
                      child: Text('Predict')),
                  const SizedBox(
                    height: 10,
                  ),
                  ElevatedButton(onPressed: pickImage, child: Text('Reselect')),
                ],
              ),
      ),
    );
  }
}
