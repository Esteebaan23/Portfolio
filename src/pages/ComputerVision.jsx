import SectionNav from "../components/SectionNav";
import ProjectCard from "../components/ProjectCard";
import CV_PROJECTS from "../data/projects.computervision";

export default function ComputerVision() {
  return (
    <div className="section domain-cv">
      <h1>Computer Vision</h1>
      <p className="sub">
        Detection, segmentation, medical imaging, evaluation workflows, deployment.
      </p>

      <SectionNav />

      <div className="featureList">
        {CV_PROJECTS.map((p) => (
          <ProjectCard key={p.title} p={p} />
        ))}
      </div>
    </div>
  );
}