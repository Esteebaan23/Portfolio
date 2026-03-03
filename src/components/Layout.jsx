import { useEffect } from "react";
import { Outlet, useLocation } from "react-router-dom";
import Header from "./Header";

export default function Layout() {
  const { pathname } = useLocation();

  useEffect(() => {
    // Fuerza top cuando cambias de página
    window.scrollTo({ top: 0, left: 0, behavior: "instant" });
  }, [pathname]);

  return (
    <div className="container">
      <Header />
      <main className="main">
        <Outlet />
      </main>
      <footer className="footer">
        © {new Date().getFullYear()} Harold E. Lucero. Built with React + Vite, deployed on GitHub Pages.
      </footer>
    </div>
  );
}