import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import "./Navbar.css";

function Navbar() {
  const location = useLocation();
  const navigate = useNavigate();
  const { t } = useTranslation();

  // دالة عامة بتتعامل مع كل اللينكات
  const handleNavClick = (e, targetId) => {
    e.preventDefault();

    if (location.pathname === "/") {
      // لو أنا بالفعل في صفحة الهوم
      const section = document.querySelector(targetId);
      if (section) {
        section.scrollIntoView({ behavior: "smooth" });
      }
    } else {
      // لو أنا في صفحة تانية (زي Upload)
      navigate("/", { state: { scrollTo: targetId } });
    }
  };

  return (
    <nav className="nav-glass">
      <ul>
        <li>
          <a href="#model" onClick={(e) => handleNavClick(e, "#model")}>
            {t("navbar.home")}
          </a>
        </li>
        <li>
          <a href="#overview" onClick={(e) => handleNavClick(e, "#overview")}>
            {t("navbar.overview")}
          </a>
        </li>
        <li>
          <a href="#home" onClick={(e) => handleNavClick(e, "#home")}>
            {t("navbar.get_started")}
          </a>
        </li>
        <li>
          <a href="#about" onClick={(e) => handleNavClick(e, "#about")}>
            {t("navbar.about_us")}
          </a>
        </li>
        <li>
          <a href="#contact" onClick={(e) => handleNavClick(e, "#contact")}>
            {t("navbar.contact")}
          </a>
        </li>
      </ul>
    </nav>
  );
}

export default Navbar;



