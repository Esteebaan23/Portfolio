import profileImg from "../assets/Harold.jpg";
import { Link, useLocation } from "react-router-dom";
import { Github, Linkedin, Mail, FileText, Home } from "lucide-react";
import { GraduationCap } from "lucide-react";
export default function Header() {
  const { pathname } = useLocation();

  return (
    <div className="nav">
      <div className="brandRow">
        <img className="avatar" src={profileImg} alt="Harold Lucero" />
        <div className="brand">
          <strong>Harold E. Lucero</strong>
          <span>Machine Learning, Computer Vision, Gen AI, Robotics</span>
        </div>
      </div>

      <div className="headerActions">

        <a href="/" className="iconBtn" title="Home">
          <Home size={18} />
        </a>

        <a href="https://github.com/Esteebaan23" target="_blank" rel="noreferrer" className="iconBtn" title="GitHub">
          <Github size={18} />
        </a>

        <a href="https://scholar.google.com/citations?hl=en&view_op=list_works&user=epB8LggAAAAJ" target="_blank" rel="noreferrer" className="iconBtn" title="Google Scholar">
          <GraduationCap  size={18} />
        </a>

        <a href="https://www.linkedin.com/in/harold-lucero-nieto-a70275259/" target="_blank" rel="noreferrer" className="iconBtn" title="LinkedIn">
          <Linkedin size={18} />
        </a>

        <a href="mailto:esteebaan.lucero.23@gmail.com" className="iconBtn" title="Email">
          <Mail size={18} />
        </a>

        <a href="https://drive.google.com/file/d/1C5iJL1vSsAKD7xPs_irfvdkySxxe_exe/view" target="_blank" rel="noreferrer" className="iconBtn" title="Recommendation Letter">
          <FileText size={18} />
        </a>

      </div>
    </div>
  );
}