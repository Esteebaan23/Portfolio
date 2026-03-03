import { Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import ComputerVision from "./pages/ComputerVision";
import NLPGenAI from "./pages/NLPGenAI";
import MLDataScience from "./pages/MLDataScience";
import Robotics from "./pages/Robotics";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Home />} />
        <Route path="/computer-vision" element={<ComputerVision />} />
        <Route path="/nlp-genai" element={<NLPGenAI />} />
        <Route path="/ml-data-science" element={<MLDataScience />} />
        <Route path="/robotics" element={<Robotics />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}