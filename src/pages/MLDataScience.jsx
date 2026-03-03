import SectionNav from "../components/SectionNav";
import ProjectCard from "../components/ProjectCard";
import MLDS_PROJECTS from "../data/projects.mldatascience";

export default function MLDataScience() {
  return (
    <div className="section domain-ml">
      <h1>ML / Data Science</h1>
      <p className="sub">
        Tabular ML, forecasting, analytics pipelines, and end-to-end experimentation with clean evaluation.
      </p>

      <SectionNav />

      <div className="featureList">
        {MLDS_PROJECTS.map((p) => (
          <ProjectCard key={p.title} p={p} />
        ))}
      </div>
    </div>
  );
}