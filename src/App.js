import React, { useEffect, useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Upload from "./pages/Upload";
import Navbar from "./components/Navbar";
import "aos/dist/aos.css";
import "./i18n";
import { useTranslation } from "react-i18next";
import { FaGlobe, FaChevronDown } from "react-icons/fa";

function App() {
  const { i18n } = useTranslation();
  const [open, setOpen] = useState(false);

  // قراءة اللغة من LocalStorage عند أول تحميل
  useEffect(() => {
    const savedLang = localStorage.getItem("appLanguage");
    if (savedLang && savedLang !== i18n.language) {
      i18n.changeLanguage(savedLang);
    }
  }, [i18n]);

  // تغيير اللغة
  const toggleLanguage = () => {
    const newLang = i18n.language === "en" ? "ar" : "en";
    i18n.changeLanguage(newLang);
    localStorage.setItem("appLanguage", newLang);
    setOpen(false);
  };

  return (
    <Router>
      {/* 🌐 زر تغيير اللغة */}
      <div
        style={{
          position: "fixed",
          top: "20px",
          left: "20px",
          zIndex: 1100,
          display: "flex",
          flexDirection: "column",
          alignItems: "flex-end",
          gap: "8px",
        }}
      >
        {/* الزرار الرئيسي */}
        <div
          onClick={() => setOpen(!open)}
          style={{
            display: "flex",
            alignItems: "center",
            gap: "10px",
            background: "rgba(255, 255, 255, 0.45)", // أبيض شفاف أنيق
            border: "1.5px solid rgba(255, 255, 255, 0.7)", // لمعة خفيفة
            borderRadius: "40px",
            padding: "8px 18px",
            cursor: "pointer",
            backdropFilter: "blur(14px)", // تأثير البلور
            transition: "all 0.3s ease",
            color: "#1e1b4b",
            boxShadow: "0 4px 10px rgba(0,0,0,0.08)",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "rgba(255, 255, 255, 0.6)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "rgba(255, 255, 255, 0.45)";
          }}
        >
          <FaGlobe size={18} color="#1e1b4b" />
          <span style={{ fontWeight: 500 }}>
            {i18n.language === "en" ? "English" : "العربية"}
          </span>
          <FaChevronDown
            size={12}
            style={{
              transform: open ? "rotate(180deg)" : "rotate(0deg)",
              transition: "transform 0.3s ease",
            }}
          />
        </div>

        {/* القائمة المنسدلة */}
        {open && (
          <div
            style={{
              background: "rgba(255, 255, 255, 0.55)",
              border: "1px solid rgba(255, 255, 255, 0.7)",
              borderRadius: "16px",
              padding: "8px 0",
              boxShadow: "0 6px 20px rgba(0,0,0,0.12)",
              backdropFilter: "blur(10px)",
              width: "130px",
              textAlign: "center",
            }}
          >
            <div
              onClick={toggleLanguage}
              style={{
                padding: "8px 0",
                cursor: "pointer",
                color: "#1e1b4b",
                fontWeight: 500,
                transition: "0.3s",
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.background = "rgba(255,255,255,0.4)")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.background = "transparent")
              }
            >
              {i18n.language === "en" ? "العربية" : "English"}
            </div>
          </div>
        )}
      </div>

      {/* 🔹 Navbar + Routes */}
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/upload" element={<Upload />} />
      </Routes>
    </Router>
  );
}

export default App;

