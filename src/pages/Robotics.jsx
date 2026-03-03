import SectionNav from "../components/SectionNav";
import ProjectCard from "../components/ProjectCard";
import ROBOTICS_PROJECTS from "../data/projects.robotics";

export default function Robotics() {
  return (
    <div className="section domain-robotics">
      <h1>Robotics</h1>
      <p className="sub">
        Mechatronics + perception + embedded integration. Vision-driven robotics and control prototypes.
      </p>

      <SectionNav />

      <div className="featureList">
        {ROBOTICS_PROJECTS.map((p) => (
          <ProjectCard key={p.title} p={p} />
        ))}
      </div>
    </div>
  );
}