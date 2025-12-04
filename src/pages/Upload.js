import React, { useState } from "react";
import "./Upload.css";
import { useTranslation } from "react-i18next";

function Upload() {
  const [selectedFile, setSelectedFile] = useState(null);
  const { t } = useTranslation();

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleUpload = () => {
    if (selectedFile) {
      alert(`${t("upload.uploading")}: ${selectedFile.name}`);
    } else {
      alert(t("upload.select_first"));
    }
  };

  const handleCancel = () => {
    setSelectedFile(null);
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
          <button className="analyze-btn" onClick={handleUpload}>
            {t("upload.analyze")}
          </button>
          <button className="cancel-btn" onClick={handleCancel}>
            {t("upload.cancel")}
          </button>
        </div>
      </div>
    </div>
  );
}

export default Upload;
