import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image, ActivityIndicator, Platform } from 'react-native';
import * as ImagePicker from 'react-native-image-picker';
import RNFS from 'react-native-fs';
import * as ort from 'onnxruntime-react-native';
import ImageResizer from 'react-native-image-resizer';
import jpeg from 'jpeg-js';
import { Buffer } from 'buffer';

global.Buffer = Buffer;

const labels = ['minor', 'moderate', 'severe'];


const App = () => {
  const [isModelReady, setIsModelReady] = useState(false);
  const [session, setSession] = useState(null);
  const [imageUri, setImageUri] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [inputName, setInputName] = useState('');

  useEffect(() => {
    const loadModel = async () => {
      try {
        const modelPath = Platform.OS === 'ios'
          ? `${RNFS.MainBundlePath}/model_clean.onnx`
          : `${RNFS.DocumentDirectoryPath}/model_clean.onnx`;

        if (Platform.OS === 'android') {
          await RNFS.copyFileAssets('model_clean.onnx', modelPath);
        }

        const session = await ort.InferenceSession.create(modelPath);
        setInputName(session.inputNames[0]);
        setSession(session);
        setIsModelReady(true);
        console.log('Model loaded');
      } catch (e) {
        console.error('Error loading model:', e);
      }
    };

    loadModel();
  }, []);

  const pickImage = () => {
    ImagePicker.launchImageLibrary(
      { mediaType: 'photo', includeBase64: false },
      async (response) => {
        if (response.didCancel || response.errorMessage) return;
        const asset = response.assets?.[0];
        if (!asset?.uri) return;

        if (!asset.fileName?.toLowerCase().endsWith('.jpg') && !asset.type?.includes('jpeg')) {
          alert("⚠️ Please choose a JPEG image.");
          return;
        }

        setImageUri(asset.uri);
        await processImage(asset.uri);
      }
    );

  };

  const processImage = async (uri) => {
    if (!session) return;
    setLoading(true);
    try {
      const inputTensor = await preprocessImage(uri);
      console.log("📷 Input tensor:", inputTensor);

      const feeds = { [inputName]: inputTensor };
      const results = await session.run(feeds);
      const outputName = session.outputNames[0];
      const outputTensor = results[outputName];

      console.log("📊 Raw output:", outputTensor.data);


      if (outputTensor.data.some(v => isNaN(v))) {
        setPrediction("Invalid prediction (NaN)");
        console.warn(" Output contains NaN");
        return;
      }

      const outputData = softmax(outputTensor.data);
      const maxIndex = outputData.indexOf(Math.max(...outputData));
      setPrediction(labels[maxIndex]);
      console.log('✅ Final Prediction:', labels[maxIndex]);

    } catch (err) {
      console.error(' Error during prediction:', err);
    } finally {
      setLoading(false);
    }
  };


  const preprocessImage = async (uri) => {
    try {
      const resized = await ImageResizer.createResizedImage(
        uri,
        256,
        256,
        'JPEG',
        100,
        0,
        undefined,
        false, // keepAspectRatio: false
        // { mode: 'stretch' } 
      );
      
  
      const base64Str = await RNFS.readFile(resized.uri, 'base64');
      const buffer = Buffer.from(base64Str, 'base64');
      const rawImageData = jpeg.decode(buffer, { useTArray: true });
  
      if (!rawImageData || !rawImageData.data) {
        throw new Error("❌ JPEG decode failed. Image might be corrupted or not a JPEG.");
      }
  
      const { width, height, data } = rawImageData;
  
  
      console.log("🖼 Resized image size:", width, "x", height);
  
      // Crop from center (this assumes width/height are 256)
      const cropX = (width - 224) / 2;
      const cropY = (height - 224) / 2;
  
      const floatData = new Float32Array(3 * 224 * 224);
      let pixelIndex = 0;
  
      for (let y = 0; y < 224; y++) {
        for (let x = 0; x < 224; x++) {
          const srcX = x + cropX;
          const srcY = y + cropY;
          const i = (srcY * width + srcX) * 4;
  
          if (i + 2 >= data.length || i < 0) {
            floatData[pixelIndex] = 0;
            floatData[pixelIndex + 224 * 224] = 0;
            floatData[pixelIndex + 2 * 224 * 224] = 0;
            pixelIndex++;
            continue;
          }
  
          const r = data[i] / 255;
          const g = data[i + 1] / 255;
          const b = data[i + 2] / 255;
  
          floatData[pixelIndex] = isNaN(r) ? 0 : r;
          floatData[pixelIndex + 224 * 224] = isNaN(g) ? 0 : g;
          floatData[pixelIndex + 2 * 224 * 224] = isNaN(b) ? 0 : b;
  
          pixelIndex++;
        }
      }
  
      return new ort.Tensor('float32', floatData, [1, 3, 224, 224]);
    } catch (err) {
      console.error(" Error in preprocessImage:", err.message);
      throw err;
    }
  };
  




  const softmax = (arr) => {
    const exp = arr.map(x => Math.exp(x));
    const sum = exp.reduce((a, b) => a + b);
    return exp.map(val => val / sum);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Rust Classifier (ONNX)</Text>
      {isModelReady ? (
        <>
          <TouchableOpacity style={styles.button} onPress={pickImage}>
            <Text style={styles.buttonText}>Pick an Image</Text>
          </TouchableOpacity>
          {imageUri && <Image source={{ uri: imageUri }} style={styles.image} />}
          {loading ? (
            <ActivityIndicator size="large" color="#007bff" />
          ) : (
            prediction && <Text style={styles.prediction}>Prediction: {prediction}</Text>
          )}
        </>
      ) : (
        <Text>Loading model...</Text>
      )}
    </View>
  );
};

export default App;

const styles = StyleSheet.create({
  container: {
    flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#fff', padding: 20,
  },
  title: {
    fontSize: 22, fontWeight: 'bold', marginBottom: 20,
  },
  button: {
    backgroundColor: '#007bff', padding: 15, borderRadius: 5,
  },
  buttonText: {
    color: '#fff', fontSize: 16,
  },
  image: {
    width: 200, height: 200, marginTop: 20,
  },
  prediction: {
    fontSize: 18, fontWeight: 'bold', marginTop: 10, color: '#333',
  },
});
