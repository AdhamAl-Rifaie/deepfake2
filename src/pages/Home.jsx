import React, { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import AOS from "aos";
import "aos/dist/aos.css";
import "./Home.css";
import { FaLinkedin, FaGithub, FaInstagram, FaFacebook } from "react-icons/fa";
import aiImage from "../assets/image2.png";
import { motion } from "framer-motion";
import { ChevronDown } from "lucide-react";
import { useTranslation } from "react-i18next";

function HomePage() {
  const { t } = useTranslation();
// بداية كود الباك
  // بنحفظ بيانات الـ form هنا
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    message: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitMessage, setSubmitMessage] = useState("");

  // دي الدالة اللي هترسل البيانات للـ backend
  const handleSubmit = async (e) => {
    e.preventDefault(); // بنمنع الـ form من إعادة تحميل الصفحة
    
    setIsSubmitting(true);
    setSubmitMessage("");

    try {
      // بنرسل البيانات للـ backend
      const API_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";
      const response = await fetch(`${API_URL}/api/contact`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (data.success) {
        // لو نجح الإرسال
        setSubmitMessage(t("contact.success"));
        // بنمسح الحقول
        setFormData({ name: "", email: "", message: "" });
      } else {
        // لو في مشكلة
        console.error("Backend error:", data.error || data.message);
        // لو الـ backend بعت كود ترجمة، نستخدمه، لو لأ نستخدم الرسالة الافتراضية
        if (data.message === "fill_fields") {
          setSubmitMessage(t("contact.fill_fields"));
        } else {
          setSubmitMessage(t("contact.error"));
        }
      }
    } catch (error) {
      console.error("Error sending message:", error);
      console.error("Error details:", error.message);
      setSubmitMessage(t("contact.connection_error"));
    } finally {
      setIsSubmitting(false);
    }
  };
// نهاية كود الباك


  // دي الدالة اللي بتحدث البيانات لما المستخدم يكتب في الحقول
  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const scrollTeam = (direction) => {
    const slider = document.getElementById("teamSlider");
    if (slider) {
      const scrollAmount = direction === "left" ? -300 : 300;
      slider.scrollBy({ left: scrollAmount, behavior: "smooth" });
    }
  };

  useEffect(() => {
    AOS.init({
      duration: 1000,
      once: true,
    });
  }, []);

  useEffect(() => {
    const sections = document.querySelectorAll(".section");
    const nav = document.querySelector(".nav-glass");

    const handleScroll = () => {
      sections.forEach((section) => {
        const rect = section.getBoundingClientRect();
        if (
          rect.top <= window.innerHeight / 2 &&
          rect.bottom >= window.innerHeight / 2
        ) {
          if (section.classList.contains("light-section")) {
            nav.classList.add("dark-nav");
          } else {
            nav.classList.remove("dark-nav");
          }
        }
      });
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const location = useLocation();
  useEffect(() => {
    if (location.state && location.state.scrollTo) {
      const section = document.querySelector(location.state.scrollTo);
      if (section) {
        section.scrollIntoView({ behavior: "smooth" });
      }
    }
  }, [location]);

  useEffect(() => {
    const section = document.querySelector(".ai-section");
    if (section) {
      setTimeout(() => {
        section.classList.add("loaded");
      }, 200);
    }
  }, []);

  useEffect(() => {
    const section = document.querySelector(".dark-section");
    if (section) {
      setTimeout(() => {
        section.classList.add("loaded");
      }, 300);
    }
  }, []);

  return (
    <div className="home-container">
      {/* Navbar */}
      <nav className="nav-glass">
        <ul>
          <li><a href="#model">{t("navbar.home")}</a></li>
          <li><a href="#overview">{t("navbar.overview")}</a></li>
          <li><a href="#home">{t("navbar.get_started")}</a></li>
          <li><a href="#about">{t("navbar.about_us")}</a></li>
          <li><a href="#contact">{t("navbar.contact")}</a></li>
        </ul>
      </nav>

      {/* Section 1 - Hero */}
      <section
        className="section dark-section"
        id="model"
        style={{ position: "relative", width: "100vw", height: "100vh", overflow: "hidden" }}
      >
        <div style={{ position: "absolute", top: "-5%", left: 0, width: "100%", height: "110%", overflow: "hidden", zIndex: 0 }}>
          <img
            src={aiImage}
            alt="AI DeepFake Visual"
            className="hero-bg-image"
            style={{ width: "100%", height: "100%", objectFit: "cover", objectPosition: "center 80%", filter: "brightness(0.9) contrast(1.1)" }}
          />
          <div style={{ position: "absolute", inset: 0, background: "linear-gradient(to bottom, rgba(0,0,0,0.15) 10%, rgba(0,0,0,0.4) 90%)" }}></div>
        </div>

        <motion.div
          initial={{ opacity: 0, x: -60 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 1.2, ease: "easeOut" }}
          style={{ position: "absolute", top: "50%", left: "8%", transform: "translateY(-50%)", color: "white", zIndex: 2 }}
        >
          <h1 style={{ fontSize: "4rem", fontWeight: "700", marginBottom: "0.7rem", textShadow: "0px 0px 15px rgba(0,255,255,0.6)" }}>
            {t("hero.title")}
          </h1>
          <p style={{ fontSize: "1.8rem", opacity: 0.85, lineHeight: "1.6", textShadow: "0px 0px 10px rgba(255, 255, 255, 0.6)", textAlign: "center" }}>
            {t("hero.subtitle")}
          </p>
        </motion.div>

        <motion.div
          animate={{ y: [0, 18, 0] }}
          transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
          style={{
            position: "absolute",
            bottom: "40px",
            left: "50%",
            transform: "translateX(-50%)",
            cursor: "pointer",
            backgroundColor: "rgba(255, 255, 255, 0.15)",
            padding: "18px",
            borderRadius: "50%",
            backdropFilter: "blur(8px)",
            boxShadow: "0 0 20px rgba(255,255,255,0.7)",
            zIndex: 2,
            border: "1px solid rgba(255,255,255,0.4)",
          }}
          whileHover={{ scale: 1.15 }}
          onClick={() => {
            const nextSection = document.getElementById("overview");
            if (nextSection) nextSection.scrollIntoView({ behavior: "smooth" });
          }}
        >
          <ChevronDown size={38} color="white" strokeWidth={2.2} />
        </motion.div>
      </section>

      {/* Section 2 - Overview */}
      <div className="overview-wrapper">
        <div className="overview-square">Box 1</div>
        <div className="overview-square">Box 2</div>
        <div className="overview-square">Box 3</div>
      </div>

      <section className="section light-section" id="overview">
        <div className="overview-container" data-aos="fade-up">
          <h2>{t("overview.title")}</h2>
          <p className="overview-intro">{t("overview.intro")}</p>

          <div className="overview-grid">
            <div className="overview-card" data-aos="fade-right">
              <h3>{t("overview.goal_title")}</h3>
              <p>{t("overview.goal_text")}</p>
            </div>

            <div className="overview-card" data-aos="zoom-in">
              <h3>{t("overview.tech_title")}</h3>
              <p>{t("overview.tech_text")}</p>
            </div>

            <div className="overview-card" data-aos="fade-left">
              <h3>{t("overview.impact_title")}</h3>
              <p>{t("overview.impact_text")}</p>
            </div>
          </div>
        </div>
      </section>

      {/* Section 3 - Get Started */}
      <section className="section dark-section" id="home">
        <div className="text-box" data-aos="fade-up" style={{ position: "relative", zIndex: 2 }}>
          <h2>{t("get_started.title")}</h2>
          <p>{t("get_started.text")}</p>
          <Link to="/upload" style={{ textDecoration: "none" }}>
            <button className="start-btn" data-aos="zoom-in" style={{ background: "#a5b4fc", color: "#fff", padding: "12px 32px", borderRadius: "30px", border: "none", fontSize: "1.1rem", cursor: "pointer", marginTop: "20px", transition: "all 0.3s ease" }}>
              {t("get_started.button")}
            </button>
          </Link>
        </div>
      </section>

      {/* Section 4 - About */}
      <section className="section light-section" id="about">
        <div className="team-container" data-aos="fade-up">
          <h2 className="team-title">{t("about.title")}</h2>
          <p className="team-subtitle">{t("about.subtitle")}</p>

          <div className="team-slider-wrapper" data-aos="fade-up">
            <button className="arrow-btn left" onClick={() => scrollTeam("left")}>&#10094;</button>
            <div className="team-slider" id="teamSlider">
              <div className="team-card" data-aos="flip-left">
                <h3>Tawfeq Mohamed</h3>
                <p>AI Engineer</p>
              </div>
              <div className="team-card" data-aos="flip-up">
                <h3>Adham Osama</h3>
                <p>AI Engineer</p>
              </div>
              <div className="team-card" data-aos="flip-right">
                <h3> Mariam Salah</h3>
                <p>Backend Developer</p>
              </div>
              <div className="team-card" data-aos="flip-left">
                <h3>Arwa Hisham</h3>
                <p>Frontend Developer</p>
              </div>
              <div className="team-card" data-aos="flip-up">
                <h3>Malak Arfa</h3>
                <p>Flutter Developer</p>
              </div>
              <div className="team-card" data-aos="flip-left">
                <h3>Basmala Hashim</h3>
                <p>Flutter Developer</p>
              </div>
            </div>
            <button className="arrow-btn right" onClick={() => scrollTeam("right")}>&#10095;</button>
          </div>
        </div>
      </section>

      {/* Section 5 - Contact */}
      <section className="section dark-section" id="contact">
        <div className="contact-wrapper" style={{ position: "relative", zIndex: 2 }}>
          <div className="contact-left" data-aos="fade-right">
            <h2>{t("contact.title")}</h2>
            <p>{t("contact.text")}</p>
            <ul className="social-list">
              <li>
                <a href="https://www.linkedin.com/in/taw7?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank" rel="noreferrer">
                  <FaLinkedin size={28} color="#a5b4fc" style={{ marginRight: "10px" }} /> {t("contact.socials.linkedin")}
                </a>
              </li>
              <li>
                <a href="https://github.com/TawfekMohamed-7" target="_blank" rel="noreferrer">
                  <FaGithub size={28} color="#a5b4fc" style={{ marginRight: "10px" }} /> {t("contact.socials.github")}
                </a>
              </li>
              <li>
                <a href="https://www.instagram.com/taww_7?igsh=aHhweXE0Z2Q1eHlp" target="_blank" rel="noreferrer">
                  <FaInstagram size={28} color="#a5b4fc" style={{ marginRight: "10px" }} /> {t("contact.socials.instagram")}
                </a>
              </li>
              <li>
                <a href="https://www.facebook.com/share/1GbmVJwrdo/" target="_blank" rel="noreferrer">
                  <FaFacebook size={28} color="#a5b4fc" style={{ marginRight: "10px" }} /> {t("contact.socials.facebook")}
                </a>
              </li>
            </ul>
          </div>

          <div className="contact-right" data-aos="fade-left" style={{ position: "relative", zIndex: 2 }}>
            <h2>{t("contact.send_message")}</h2>
            <form className="contact-form" onSubmit={handleSubmit}>
              <input
                id="contact-name"
                type="text"
                name="name"
                placeholder={t("contact.name_placeholder")}
                value={formData.name}
                onChange={handleChange}
                required
              />
              <input
                id="contact-email"
                type="email"
                name="email"
                placeholder={t("contact.email_placeholder")}
                value={formData.email}
                onChange={handleChange}
                required
              />
              <textarea
                id="contact-message"
                name="message"
                placeholder={t("contact.message_placeholder")}
                rows="5"
                value={formData.message}
                onChange={handleChange}
                required
              ></textarea>
              <button
                type="submit"
                className="contact-btn"
                disabled={isSubmitting}
              >
                {isSubmitting ? t("contact.sending") : t("contact.submit_button")}
              </button>
              {submitMessage && (
                <p style={{
                  marginTop: "15px",
                  padding: "10px",
                  borderRadius: "5px",
                  backgroundColor: submitMessage === t("contact.success") ? "rgba(76, 175, 80, 0.2)" : "rgba(244, 67, 54, 0.2)",
                  color: submitMessage === t("contact.success") ? "#4caf50" : "#f44336",
                  textAlign: "center",
                  fontSize: "0.9rem"
                }}>
                  {submitMessage}
                </p>
              )}
            </form>
          </div>
        </div>
      </section>
    </div>
  );
}

export default HomePage;
