// src/components/Prediction.js

import React, { useState } from 'react';

const Prediction = () => {
  const [prediction, setPrediction] = useState(null);

  const handlePredict = () => {
    // Handle the logic for getting the prediction (fetch from backend)
    console.log('Making prediction...');
    // Simulate a prediction result
    setPrediction('Moderate Severity');
  }

  return (
    <div>
      <h2>Prediction</h2>
      <button onClick={handlePredict}>Get Prediction</button>
      {prediction && (
        <div>
          <h3>Prediction Result:</h3>
          <p>{prediction}</p>
        </div>
      )}
    </div>
  );
}

export default Prediction;
