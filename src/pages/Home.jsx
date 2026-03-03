import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import profileImg from "../assets/Harold.jpg";

import PUBLICATIONS from "../data/publications";
import AWARDS from "../data/awards";
import CERTIFICATES from "../data/certificates";

function useCountUp(target, durationMs = 900) {
  const [value, setValue] = useState(0);

  useEffect(() => {
    let raf = 0;
    const start = performance.now();

    const tick = (t) => {
      const p = Math.min((t - start) / durationMs, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setValue(Math.round(eased * target));
      if (p < 1) raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, durationMs]);

  return value;
}

export default function Home() {
  const sectionIds = useMemo(
    () => [
      { id: "about", label: "About" },
      { id: "education", label: "Education" },
      { id: "awards", label: "Awards" },
      { id: "publications", label: "Publications" },
      { id: "certs", label: "Certificates" },
    ],
    []
  );

  const [active, setActive] = useState("about");
  const refs = useRef({});

  // Ajusta si quieres
  const statYears = useCountUp(2);
  const statProjects = useCountUp(17);
  const statDomains = useCountUp(4);

  const scrollTo = (id) => {
    const el = refs.current[id];
    if (!el) return;

    const headerOffset = 110; // ajusta: 90–130 según tu header
    const y = el.getBoundingClientRect().top + window.pageYOffset - headerOffset;

    window.scrollTo({ top: y, behavior: "smooth" });
  };

  useEffect(() => {
    const els = sectionIds.map((s) => refs.current[s.id]).filter(Boolean);
    if (els.length === 0) return;

    const obs = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => (b.intersectionRatio ?? 0) - (a.intersectionRatio ?? 0))[0];

        if (visible?.target?.id) setActive(visible.target.id);
      },
      { threshold: [0.2, 0.35, 0.5], rootMargin: "-18% 0px -60% 0px" }
    );

    els.forEach((el) => obs.observe(el));
    return () => obs.disconnect();
  }, [sectionIds]);

  return (
    <div className="homeLayout">
      {/* LEFT: Table of contents */}
      <aside className="toc">
        <div className="tocTitle">
          <span className="tocKicker">Table of contents</span>
          <span className="tocArrow">↘</span>
        </div>

        <div className="tocGroup">
          <div className="tocGroupTitle">Contents</div>
          {sectionIds.map((s) => (
            <button
              key={s.id}
              className={`tocItem toc-${s.id} ${active === s.id ? "active" : ""}`}
              onClick={() => scrollTo(s.id)}
              type="button"
            >
              {s.label}
            </button>
          ))}
        </div>

        <div className="tocCTA">
          <div className="tocCTATitle">Explore by domain</div>
          <div className="tocButtons">
            <Link className="btn domain-cv" to="/computer-vision">Computer Vision</Link>
            <Link className="btn domain-nlp" to="/nlp-genai">NLP / GenAI</Link>
            <Link className="btn domain-ml" to="/ml-data-science">ML / Data Science</Link>
            <Link className="btn domain-robotics" to="/robotics">Robotics</Link>
          </div>
        </div>
      </aside>

      {/* RIGHT: Content */}
      <div className="homeMain">
        {/* HERO */}
        <section className="homeHeroStack" id="about" ref={(el) => (refs.current.about = el)}>
          {/* Big round avatar */}
          <div className="heroAvatarWrap">
            <div className="heroAvatarNeon">
              <img src={profileImg} alt="Harold Lucero portrait" />
            </div>

            <div className="heroCaption">
              Machine Learning Engineer
            </div>
          </div>

          <div className="heroCopy">
            <h1 className="h1">Harold E. Lucero</h1>

            <div className="heroBadges">
              <span className="pill">Machine Learning</span>
              <span className="pill">Computer Vision</span>
              <span className="pill">NLP / GenAI</span>
              <span className="pill">Robotics</span>
            </div>

            <p className="sub">
              Machine Learning Engineer with 2+ years of hands-on experience designing end-to-end AI systems across Computer Vision, Gen AI, and Robotics. 
              Combines a Master’s degree in Artificial Intelligence with a strong foundation in Mechatronics Engineering to build intelligent solutions.
            </p>

            <p className="sub">
              Designed hybrid deep learning architectures for medical imaging, developed object detection and perception systems
              for autonomous platforms, and implemented RAG assistants using vector databases and
              large language models. Experience spans data engineering, model optimization, evaluation pipelines, and cloud deployment
              workflows, with a focus on reliability and real-world constraints.
            </p>

            <div className="statsRow">
              <div className="statCard">
                <div className="statNum">{statYears}+</div>
                <div className="statLabel">Years in ML</div>
              </div>
              <div className="statCard">
                <div className="statNum">{statProjects}+</div>
                <div className="statLabel">Projects</div>
              </div>
              <div className="statCard">
                <div className="statNum">{statDomains}+</div>
                <div className="statLabel">Domains</div>
              </div>
            </div>

            <div className="heroActions">
              <Link className="btn" to="/computer-vision">View Projects</Link>
            </div>
          </div>
        </section>

        {/* EDUCATION */}
        <section className="homeSectionCard section-education" id="education" ref={(el) => (refs.current.education = el)}>
          
          <h2>Education</h2>
          <div className="miniGrid">
            <div className="miniCard">
              <div className="miniTitle">M.S. in Artificial Intelligence</div>
              <div className="miniMeta">University of North Texas</div>
            </div>
            <div className="miniCard">
              <div className="miniTitle">B.S. in Mechatronics Engineering</div>
              <div className="miniMeta">Mechatronics Engineering</div>
            </div>
          </div>
        </section>

        {/* AWARDS */}
        <section className="homeSectionCard section-awards" id="awards" ref={(el) => (refs.current.awards = el)}>
          <h2>Awards</h2>
          <div className="miniGrid">
            {AWARDS.map((a) => (
              <div className="miniCard" key={a.title}>
                <div className="miniTitle">{a.title}</div>
                {a.description && <div className="miniText">{a.description}</div>}
                {a.link && (
                  <div className="links">
                    <a className="taglink" href={a.link} target="_blank" rel="noreferrer">
                      {a.linkLabel || "View"}
                    </a>
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>

        {/* PUBLICATIONS */}
        <section className="homeSectionCard section-publications" id="publications"  ref={(el) => (refs.current.publications = el)}>
          <h2>Publications</h2>
          <div className="miniGrid">
            {PUBLICATIONS.map((p) => (
              <div className="miniCard" key={p.title}>
                <div className="miniTitle">{p.title}</div>
                {(p.venue || p.year) && (
                  <div className="miniMeta">{[p.venue, p.year].filter(Boolean).join(" • ")}</div>
                )}
                {p.description && <div className="miniText">{p.description}</div>}
                {p.link && (
                  <div className="links">
                    <a className="taglink" href={p.link} target="_blank" rel="noreferrer">
                      {p.linkLabel || "View"}
                    </a>
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>

        

        {/* CERTIFICATES */}
        <section className="homeSectionCard section-certs" id="certs" ref={(el) => (refs.current.certs = el)}>
          <h2>Certificates</h2>
          <div className="miniGrid">
            {CERTIFICATES.map((c) => (
              <div className="miniCard" key={c.title}>
                <div className="miniTitle">{c.title}</div>
                {c.issuer && <div className="miniMeta">{c.issuer}</div>}
                {c.link && (
                  <div className="links">
                    <a className="taglink" href={c.link} target="_blank" rel="noreferrer">
                      {c.linkLabel || "View"}
                    </a>
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}