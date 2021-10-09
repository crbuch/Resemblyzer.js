# Resemblyzer.js
An implementation of CorentinJ's [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) in ort.js (onnxruntime) and tensorflow.js for the web. 

# What it can be used for
Given a few seconds of speech it creates a summary vector of 256 values known as an encoding. This can be used in many things such as speaker verification, deepfake detection, voice cloning, speaker diarization, and much more. The pretrained model came from [the original repo](https://github.com/resemble-ai/Resemblyzer) and was converted to onnx to use with onnxjs. I rewrote the neccessary preprocessing parts in javascript and used functions from [Magenta.js](https://github.com/magenta/magenta-js) to convert the audio to mels for the network. 
# Architecture
The network gets fed partial mels from each audio file and averages the embeddings of them after it goes through 3 lstm layers and a fully connected layer. According to the original repo, it works around 1000x real-time on gpu.

# Example
The projection of resemblyzer in python v.s. the projection of resemblyzer.js (each speaker has 10 utterances)


Resemblyzer python            |  Resemblyzer.js
:-------------------------:|:-------------------------:
![pytorch](https://EncoderMin.cooperbuch.repl.co/originalMels.png)| ![js](https://EncoderMin.cooperbuch.repl.co/jsMels.png)



# Code example
You need to import tensorflow and onnxruntime. You also need the resemblyzer.min.js and the pretrained.onnx files in a folder called "Resemblyzer" in the parent directory of your website.
```html
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width">
    <title>resemblyzer</title>
    <link href="style.css" rel="stylesheet" type="text/css" />
  </head>
  <body>
    
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
    <script src="Resemblyzer/resemblyzer.min.js"></script>
    
  </body>
</html>
```
Javascript:
```javascript
embed_audio("example_sentence.wav").then(function(t){
  //t is a tensor with 256 values
	t.print()
}


```



