import React, { useState } from "react";
import "./Upload.css";
import { useTranslation } from "react-i18next";
import axios from "axios";

function Upload() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const { t } = useTranslation();

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setPrediction(null); // Reset previous results
    setError(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert(t("upload.select_first"));
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post(
        "http://localhost:8000/predict",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          timeout: 60000, // 60 second timeout for video processing
        }
      );
      setPrediction(response.data);
      console.log("Prediction received:", response.data);
    } catch (err) {
      if (err.code === 'ECONNABORTED') {
        setError("Request timeout. Video processing took too long.");
      } else if (err.response) {
        setError(err.response?.data?.detail || "Prediction failed. Please try again.");
      } else if (err.request) {
        setError("Cannot connect to server. Make sure the backend is running on http://localhost:8000");
      } else {
        setError("An unexpected error occurred.");
      }
      console.error("Upload error:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    setSelectedFile(null);
    setPrediction(null);
    setError(null);
  };

  // Helper function to get result color
  const getResultColor = (result) => {
    return result === "Real" ? "#4caf50" : "#f44336";
  };

  // Helper function to format confidence bar
  const getConfidenceBarColor = (result) => {
    return result === "Real" ? "#4caf50" : "#f44336";
  };

  return (
    <div className="upload-page">
      <div className="upload-container">
        <h1 className="title">{t("upload.title")}</h1>
        <h2 className="subtitle">{t("upload.subtitle")}</h2>
        <p className="description">{t("upload.description")}</p>

        <div className="upload-box">
          <label htmlFor="videoUpload" className="upload-label">
            {selectedFile ? (
              <span className="file-name">{selectedFile.name}</span>
            ) : (
              <>
                <i className="fa-solid fa-cloud-arrow-up upload-icon"></i>
                <span>{t("upload.click_upload")}</span>
              </>
            )}
          </label>
          <input
            type="file"
            id="videoUpload"
            accept="video/*"
            onChange={handleFileChange}
            hidden
          />
        </div>

        <div className="buttons">
          <button
            className="analyze-btn"
            onClick={handleUpload}
            disabled={loading || !selectedFile}
          >
            {loading ? (
              <>
                <i className="fa-solid fa-spinner fa-spin"></i> Processing...
              </>
            ) : (
              t("upload.analyze")
            )}
          </button>
          <button className="cancel-btn" onClick={handleCancel}>
            {t("upload.cancel")}
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="error-message" style={{ 
            color: '#f44336', 
            backgroundColor: '#ffebee',
            padding: '15px',
            borderRadius: '8px',
            marginTop: '20px',
            border: '1px solid #f44336'
          }}>
            <p style={{ margin: 0 }}>
              <strong>❌ Error:</strong> {error}
            </p>
          </div>
        )}

        {/* Prediction Result Display */}
        {prediction && (
          <div className="result-container" style={{
            marginTop: '30px',
            padding: '25px',
            backgroundColor: '#f5f5f5',
            borderRadius: '12px',
            border: `3px solid ${getResultColor(prediction.result)}`,
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
          }}>
            <h3 style={{ 
              color: getResultColor(prediction.result),
              marginBottom: '20px',
              fontSize: '24px'
            }}>
              🎯 Detection Result
            </h3>
            
            <div style={{ marginBottom: '15px' }}>
              <p style={{ fontSize: '18px', marginBottom: '10px' }}>
                <strong>Classification:</strong> 
                <span style={{ 
                  color: getResultColor(prediction.result),
                  fontSize: '22px',
                  marginLeft: '10px',
                  fontWeight: 'bold'
                }}>
                  {prediction.result}
                </span>
              </p>
              
              <p style={{ fontSize: '16px', marginBottom: '10px' }}>
                <strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(2)}%
              </p>
              
              {/* Confidence Bar */}
              <div style={{
                width: '100%',
                height: '20px',
                backgroundColor: '#e0e0e0',
                borderRadius: '10px',
                overflow: 'hidden',
                marginBottom: '15px'
              }}>
                <div style={{
                  width: `${prediction.confidence * 100}%`,
                  height: '100%',
                  backgroundColor: getConfidenceBarColor(prediction.result),
                  transition: 'width 0.3s ease'
                }}></div>
              </div>
            </div>

            {/* Detailed Probabilities */}
            {prediction.prob_real !== undefined && prediction.prob_fake !== undefined && (
              <div style={{ 
                marginTop: '20px',
                padding: '15px',
                backgroundColor: 'white',
                borderRadius: '8px'
              }}>
                <h4 style={{ marginBottom: '10px', fontSize: '16px' }}>📊 Detailed Analysis:</h4>
                <p style={{ margin: '5px 0' }}>
                  <strong>Real Probability:</strong> {(prediction.prob_real * 100).toFixed(2)}%
                </p>
                <p style={{ margin: '5px 0' }}>
                  <strong>Fake Probability:</strong> {(prediction.prob_fake * 100).toFixed(2)}%
                </p>
              </div>
            )}
            
            {prediction.filename && (
              <p style={{ 
                marginTop: '15px',
                fontSize: '14px',
                color: '#666'
              }}>
                <strong>📁 File:</strong> {prediction.filename}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default Upload;
