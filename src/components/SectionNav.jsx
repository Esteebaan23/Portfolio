import { Link } from "react-router-dom";

export default function SectionNav({ showHome = true }) {
  return (
    <div className="sectionNav">
      {showHome && (
        <Link className="btn nav-home" to="/">
          Home
        </Link>
      )}

      <Link className="btn nav-cv" to="/computer-vision">Computer Vision</Link>
      <Link className="btn nav-nlp" to="/nlp-genai">NLP / GenAI</Link>
      <Link className="btn nav-ml" to="/ml-data-science">ML / Data Science</Link>
      <Link className="btn nav-robotics" to="/robotics">Robotics</Link>
    </div>
  );
}