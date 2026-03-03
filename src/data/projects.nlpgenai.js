import ragpdf from "../assets/pdf_RAG.png"; 
import ragBilingual from "../assets/RAG.jpg"; 
import disaster from "../assets/disaster.png";

const NLP_PROJECTS = [
  {
    title: "Resume Analyzer RAG Assistant",
    meta: "LangChain, FAISS, PDF parsing, embeddings, Gradio",
    image: ragpdf,
    alt: "Resume Analyzer RAG assistant preview",
    bullets: [
      "Developed a Retrieval-Augmented Generation (RAG) assistant using LangChain and FAISS to generate document-grounded answers and automated summaries from private PDFs.",
      "Implemented an end-to-end pipeline (PDF parsing → chunking → embeddings → FAISS vector indexing → semantic retrieval → answer generation) to extract key insights from documents efficiently.",
      "Built a multi-user, session-safe Gradio app to support scalable interactive Q\&A, enabling fast querying of internal documents such as reports, policies, and manuals."
    ],
    links: [
      // Cambia a tu repo real si ya existe
      { label: "GitHub", href: "https://github.com/Esteebaan23/CV-Summarize_RAG" }
    ],
  },
  {
    title: "Bilingual RAG Assistant for International Students",
    meta: "English/Spanish, RAG, vector search, fine-tuning roadmap, Streamlit",
    image: ragBilingual,
    alt: "Bilingual RAG assistant preview",
    bullets: [
      "Built a bilingual (English/Spanish) Retrieval-Augmented Generation system to answer university and international student queries using a LLaMA-based language model.",
      "Implemented semantic retrieval with FAISS and dense multilingual embeddings to deliver consistent and accurate responses across languages.",
      "Designed prompt and generation constraints to enforce single-language outputs and reduce hallucinations in high-stakes academic and immigration-related questions.",
      "Evaluated the end-to-end RAG pipeline through structured testing to ensure response accuracy, language consistency, and robustness to ambiguous queries."
    ],
    links: [
      { label: "GitHub", href: "https://github.com/Esteebaan23/Bilingual-RAG-Chatbot-for-International-Students" },
    ],
  },

  {
    title: "Disaster Tweets Classification",
    meta: "NLP text classification, LSTM and Transformer models",
    image: disaster,
    bullets: [
      "Built a text classification pipeline to identify disaster-related tweets using LSTM and BERT-based models.",
      "Implemented data preprocessing, training, and evaluation workflows using Python, PyTorch, TensorFlow, and Hugging Face.",
      "Evaluated model performance on held-out data, achieving an accuracy of 81% across both approaches."
    ],
    links: [{ label: "GitHub", href: "https://github.com/Esteebaan23/Disaster_Tweets_LSTM_Bert_Deployment" }],
  },

];

export default NLP_PROJECTS;