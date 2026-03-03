export default function ProjectCard({ p }) {
  return (
    <article className="featureCard">
      <div className="featureText">
        <h3>{p.title}</h3>
        <p className="meta">{p.meta}</p>

        {p.bullets?.length > 0 && (
          <ul>
            {p.bullets.map((b, i) => (
              <li key={i}>{b}</li>
            ))}
          </ul>
        )}

        {p.links?.length > 0 && (
          <div className="links">
            {p.links.map((l) => (
              <a key={l.label} className="taglink" href={l.href} target="_blank" rel="noreferrer">
                {l.label}
              </a>
            ))}
          </div>
        )}
      </div>

      {p.image && (
        <div className="featureMedia">
          <img src={p.image} alt={p.alt || p.title} />
        </div>
      )}
    </article>
  );
}