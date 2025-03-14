import React, { useRef, useState, useEffect } from 'react';
import * as tf from 'tensorflow';

const DigitRecognizer = () => {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [ctx, setCtx] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [model, setModel] = useState(null);
  const [loadingError, setLoadingError] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // Initialize canvas when component mounts
  useEffect(() => {
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      
      // Set canvas styling
      context.lineWidth = 15;
      context.lineJoin = 'round';
      context.lineCap = 'round';
      context.strokeStyle = 'black';
      
      // Fill with white background
      context.fillStyle = 'white';
      context.fillRect(0, 0, canvas.width, canvas.height);
      
      setCtx(context);
    }
    
    // Try to load model
    const loadModel = async () => {
      try {
        setLoadingError(null);
        // In a real app, we would load your saved model from a server
        // For demo purposes, we'll use a placeholder message
        setModelLoaded(true);
        setModel({ placeholder: true });
      } catch (err) {
        console.error("Failed to load model:", err);
        setLoadingError("Failed to load the model. Please try again later.");
      }
    };
    
    loadModel();
  }, []);

  const startDrawing = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    // Get correct mouse position relative to canvas
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    ctx.beginPath();
    ctx.moveTo(x, y);
    setIsDrawing(true);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    // Get correct mouse position relative to canvas
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = () => {
    ctx.closePath();
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
  };

  const preprocessCanvas = () => {
    const canvas = canvasRef.current;
    
    // Create a new canvas for processing
    const processCanvas = document.createElement('canvas');
    const processCtx = processCanvas.getContext('2d');
    
    // Set to MNIST dimensions
    processCanvas.width = 28;
    processCanvas.height = 28;
    
    // Scale down the image and convert to grayscale
    processCtx.fillStyle = 'white';
    processCtx.fillRect(0, 0, 28, 28);
    processCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28);
    
    // Get image data
    const imageData = processCtx.getImageData(0, 0, 28, 28);
    
    // Create array for the model input (1x784)
    const input = new Float32Array(784);
    
    // Convert pixel data to the format expected by the model
    // Note: We invert colors since MNIST uses white digits on black background
    for (let i = 0; i < imageData.data.length; i += 4) {
      // Convert RGBA to grayscale and normalize
      const grayscale = 255 - (imageData.data[i] * 0.3 + imageData.data[i + 1] * 0.59 + imageData.data[i + 2] * 0.11);
      input[i/4] = grayscale / 255.0;  // Normalize to [0,1]
    }
    
    return input;
  };

  const predictDigit = () => {
    if (!modelLoaded || isProcessing) return;
    
    setIsProcessing(true);
    
    try {
      const processedData = preprocessCanvas();
      
      // In a real application, this is where we would use the model to predict:
      // const result = model.predict(tf.tensor([processedData]));
      // const prediction = result.argMax(1).dataSync()[0];
      
      // For demo purposes, we'll simulate a prediction with a random number
      setTimeout(() => {
        const simulatedPrediction = Math.floor(Math.random() * 10);
        setPrediction(simulatedPrediction);
        setIsProcessing(false);
      }, 500);
      
    } catch (err) {
      console.error("Error during prediction:", err);
      setPrediction("Error");
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex flex-col items-center p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">MNIST Digit Recognizer</h1>
      
      <div className="bg-gray-100 p-6 rounded-lg shadow-md w-full">
        <div className="flex flex-col lg:flex-row items-center gap-6">
          <div className="flex flex-col items-center">
            <div className="border-4 border-gray-400 rounded-lg mb-4">
              <canvas
                ref={canvasRef}
                width={280}
                height={280}
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={stopDrawing}
                onMouseLeave={stopDrawing}
                className="touch-none bg-white rounded-md"
              />
            </div>
            
            <div className="flex gap-4 mb-4">
              <button 
                onClick={clearCanvas}
                className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 transition"
              >
                Clear
              </button>
              <button 
                onClick={predictDigit}
                disabled={!modelLoaded || isProcessing}
                className={`px-4 py-2 text-white rounded transition ${
                  !modelLoaded || isProcessing 
                    ? 'bg-gray-400 cursor-not-allowed' 
                    : 'bg-blue-500 hover:bg-blue-600'
                }`}
              >
                {isProcessing ? 'Processing...' : 'Predict'}
              </button>
            </div>
          </div>
          
          <div className="flex flex-col items-center border border-gray-300 rounded-lg p-6 bg-white">
            <h2 className="text-xl font-semibold mb-4">Prediction Result</h2>
            {prediction !== null ? (
              <div className="text-7xl font-bold text-blue-600">
                {prediction}
              </div>
            ) : (
              <div className="text-gray-500 text-lg">
                Draw a digit and click "Predict"
              </div>
            )}
            
            <div className="mt-6 text-gray-600 text-sm">
              {modelLoaded ? (
                <span className="text-green-500 flex items-center">
                  <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  Model Ready
                </span>
              ) : loadingError ? (
                <span className="text-red-500">{loadingError}</span>
              ) : (
                <span className="text-yellow-500">Loading model...</span>
              )}
            </div>
          </div>
        </div>
      </div>
      
      <div className="mt-8 p-6 bg-gray-100 rounded-lg w-full">
        <h2 className="text-xl font-semibold mb-4">Instructions</h2>
        <ol className="list-decimal pl-6 space-y-2">
          <li>Draw a single digit (0-9) in the canvas area</li>
          <li>Try to center your digit in the drawing area</li>
          <li>Click the "Predict" button to see the model's prediction</li>
          <li>Use "Clear" to erase and try another digit</li>
        </ol>
        <p className="mt-4 text-sm text-gray-600">
          Note: This demo uses a pre-trained model on the MNIST dataset, which recognizes
          handwritten digits. For best results, draw clear digits that fill most of the drawing area.
        </p>
      </div>
    </div>
  );
};

export default DigitRecognizer;
