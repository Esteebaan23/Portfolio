import SectionNav from "../components/SectionNav";
import ProjectCard from "../components/ProjectCard";
import NLP_PROJECTS from "../data/projects.nlpgenai";

export default function NLPGenAI() {
  return (
    <div className="section domain-nlp">
      <h1>NLP / GenAI</h1>
      <p className="sub">
        RAG assistants, LLM apps, document understanding, bilingual pipelines, and retrieval evaluation.
      </p>

      <SectionNav />

      <div className="featureList">
        {NLP_PROJECTS.map((p) => (
          <ProjectCard key={p.title} p={p} />
        ))}
      </div>
    </div>
  );
}