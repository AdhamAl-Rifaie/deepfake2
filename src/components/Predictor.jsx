import axios from "axios";
import { useState } from "react";

function Predictor() {
  const [features, setFeatures] = useState("");
  const [prediction, setPrediction] = useState(null);

  const handlePredict = async () => {
    const arr = features.split(",").map(v => parseFloat(v.trim()));
    const res = await axios.post("http://localhost:8000/predict", {
      features: arr,
    });
    setPrediction(res.data.prediction);
  };

  return (
    <div>
      <input
        value={features}
        onChange={e => setFeatures(e.target.value)}
        placeholder="1.2, 3.4, 5.6"
      />
      <button onClick={handlePredict}>Predict</button>
      {prediction !== null && <p>Prediction: {prediction}</p>}
    </div>
  );
}

export default Predictor;
