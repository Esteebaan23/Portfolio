import ragImg from "./assets/rag.jpg"
import organoidsImg from "./assets/organoids.png"
import smallObjectsImg from "./assets/small-objects.png"
import robotYoloImg from "./assets/robot-yolo.png"
import profileImg from "./assets/Harold.jpg"

const FEATURED = [
  {
    title: "Bilingual RAG QA Bot for International Students",
    meta: "Retrieval-Augmented Generation, multilingual NLP",
    image: ragImg,
    alt: "RAG QA Bot preview",
    bullets: [
      "Built a bilingual (English/Spanish) Retrieval-Augmented Generation system to answer university and international student queries using a LLaMA-based language model.",
      "Implemented semantic retrieval with FAISS and dense multilingual embeddings to deliver consistent and accurate responses across languages.",
      "Designed prompt and generation constraints to enforce single-language outputs and reduce hallucinations in high-stakes academic and immigration-related questions.",
      "Evaluated the end-to-end RAG pipeline through structured testing to ensure response accuracy, language consistency, and robustness to ambiguous queries."
    ],
    links: [
      { label: "GitHub", href: "https://github.com/Esteebaan23/Bilingual-RAG-Chatbot-for-International-Students" },
      //{ label: "Demo", href: "#" },
    ],
  },
  {
    title: "Organoids: Chamber Forming vs Non-Forming and Fluorescence Prediction",
    meta: "Biomedical computer vision, classification and image-to-image generation",
    image: organoidsImg,
    alt: "Organoids project preview",
    bullets: [
      "Built hybrid image classifiers combining ResNet50 and Vision Transformer architectures to distinguish chamber-forming vs non-forming organoids, achieving 95.09% accuracy and outperforming baseline models.",
      "Developed conditional image-to-image generation models using U-Net and ResUNet to predict fluorescence channels from brightfield images, reaching PSNR 24.84 and SSIM 0.928 with ResUNet.",
      "Designed an end-to-end pipeline integrating training, evaluation, and model comparison to support robust biomedical image analysis workflows.",
      "Implemented systematic evaluation across classification and generation tasks using metrics such as accuracy, precision, recall, F1-score, PSNR, and SSIM to ensure reproducibility and stable performance."
    ],
    links: [
      { label: "GitHub", href: "https://github.com/Esteebaan23/Organoid-Brightfield-to-Fluorescence-Image-Translation" },
    ],
  },
  {
    title: "D2SO: Detecting Distant and Small Objects for Vision-Based Autonomous Systems",
    meta: "Small object detection and segmentation, autonomous driving vision",
    image: smallObjectsImg,
    alt: "Small object detection preview",
    bullets: [
      "Built and optimized detection and segmentation pipelines for distant and small objects using YOLOv11, U-Net, and Meta’s SAM.",
      "Evaluated model performance on multi-class perception tasks, achieving 72.49% accuracy and 86.13% precision across humans, vehicles, and structural elements.",
      "Conducted systematic training and evaluation to improve robustness of vision-based perception in autonomous driving scenarios."
    ],
    links: [
      { label: "GitHub", href: "https://github.com/Esteebaan23/Vehicle_Human_Object_Segmentation" },
      { label: "IEEE Publication", href: "https://ieeexplore.ieee.org/document/11071446" },
    ],
  },
  {
    title: "Restricted Area Sign Detector Using YOLOv5",
    meta: "Real-time object detection, robotics integration",
    image: robotYoloImg,
    alt: "YOLOv5 robot preview",
    bullets: [
      "Built an autonomous mobile robot integrating mechanical, electronic, and software components for real-world navigation tasks.",
      "Implemented embedded control and communication pipelines using Arduino and Raspberry Pi 4 to support autonomous operation and system coordination.",
      "Developed a YOLOv5-based vision module to detect restricted-area signs and prevent unauthorized crossings, achieving 93% detection confidence."
    ],
    links: [
      { label: "YouTube Demo", href: "https://www.youtube.com/watch?v=JE850tpixUE" },
      { label: "IEEE Publication", href: "https://ieeexplore.ieee.org/document/10405395" },
    ],
  },
]

const ML_PROJECTS = [
    {
      title: "Car Driving Lane Segmentation",
      meta: "Semantic segmentation, computer vision",
      bullets: [
        "Built a lane segmentation pipeline to extract drivable lane markings from road scenes using SAM2.",
        "Implemented training and inference workflows that support rapid iteration and visual inspection of segmentation results.",
        "Generated qualitative visualizations to analyze model behavior and highlight common failure modes."
      ],
      links: [{ label: "GitHub", href: "https://github.com/Esteebaan23/Car_Driving_Lane_Segmentation_SAM2" }],
    },
    {
    title: "Classification of Licence Plates",
    meta: "Computer vision classification, model benchmarking",
    bullets: [
      "Built a license plate state classification pipeline using multiple deep learning architectures, including ResNet50, DenseNet121, VGG16, Vision Transformer, and a custom CNN.",
      "Benchmarked individual models and implemented a stacking ensemble, improving accuracy from 92.47% (best single model) to 93.52%.",
      "Conducted F1-score and error analysis to assess model robustness and validate classification performance."
    ],
    links: [{ label: "GitHub", href: "https://github.com/Esteebaan23/Classification-of-License-Plates-by-State" }],
  },
  {
    title: "Skin Cancer Classification",
    meta: "Medical imaging classification, deep learning",
    bullets: [
      "Built a medical image classification pipeline to distinguish malignant vs. benign skin lesions using multiple CNN architectures.",
      "Trained and compared ResNet50, DenseNet121, VGG16, and InceptionV3 models with consistent preprocessing and evaluation protocols.",
      "Applied image enhancement techniques such as CLAHE to improve feature representation, achieving 89% accuracy with ResNet50 + CLAHE."
    ],
    links: [{ label: "GitHub", href: "https://github.com/Esteebaan23/Skin-Cancer-Classification-Bening_vs_Malignant" }],
  },
  {
    title: "Face Mask Detection",
    meta: "Computer vision classification, real-time inference",
    bullets: [
      "Built a real-time face mask classification pipeline using OpenCV, MediaPipe for face detection, and MobileNetV2 for mask recognition.",
      "Integrated face localization with a MobileNetV2 classifier, achieving F1-scores of 96% for the “With Mask” class and 94% for the “Without Mask” class.",
      "Implemented a live visualization loop to inspect predictions and system behavior during real-time inference."
    ],
    links: [{ label: "GitHub", href: "https://github.com/Esteebaan23/Face-Mask-Detector" }],
  },
  {
    title: "House Price Prediction",
    meta: "Regression, feature engineering, model comparison",
    bullets: [
      "Built an end-to-end regression pipeline for housing price prediction, including data cleaning, encoding, and feature preparation.",
      "Applied feature engineering and multicollinearity analysis to improve XGBoost and Random Forest regression performance.",
      "Reduced RMSE by over $9,000 and improved R² by more than 4% through iterative model refinement and evaluation."
    ],
    links: [{ label: "GitHub", href: "https://github.com/Esteebaan23/House_Price_Prediction" }],
  },

  
  {
    title: "Student Performance Analysis",
    meta: "Data analysis and supervised learning, interpretable models",
    bullets: [
      "Built a predictive workflow to analyze student academic performance using supervised learning and clustering techniques.",
      "Performed exploratory data analysis and preprocessing, training Linear Regression and Random Forest models with accuracies of 88% and 87%.",
      "Trained and tuned a neural network model, achieving 98.5% accuracy and identifying key factors influencing academic outcomes."
    ],
    links: [{ label: "GitHub", href: "https://github.com/Esteebaan23/Student-Performance-Factors-Analysis" }],
  },
  {
    title: "Disaster Tweets Classification",
    meta: "NLP text classification, LSTM and Transformer models",
    bullets: [
      "Built a text classification pipeline to identify disaster-related tweets using LSTM and BERT-based models.",
      "Implemented data preprocessing, training, and evaluation workflows using Python, PyTorch, TensorFlow, and Hugging Face.",
      "Evaluated model performance on held-out data, achieving an accuracy of 81% across both approaches."
    ],
    links: [{ label: "GitHub", href: "https://github.com/Esteebaan23/Disaster_Tweets_LSTM_Bert_Deployment" }],
  },

   {
    title: "Store Sales Forecasting",
    meta: "Time Series Forecasting, Regression, XGBoost",
    bullets: [
      "Designed a time series forecasting workflow combining SARIMA baselines with gradient-boosted regression models.",
      "Trained and optimized an XGBoost regressor, reaching an RMSE of 206.54 on held-out data.",
      "Created a lightweight local visualization tool to inspect forecast trends and model behavior."
    ],
    links: [{ label: "GitHub", href: "https://github.com/Esteebaan23/Store_Sales_Time_Series_XGboost" }],
  },
]

const AWARDS = [
  {
    title: "Distinguished Artificial Intelligence Student",
    description:
      "Awarded for consistent academic excellence and high-impact contributions across coursework and applied engineering projects.",
    link: "https://drive.google.com/file/d/1d3mRcob3gmt7jy9i-auv6ML1ukVkQ75g/view?usp=sharing",
    linkLabel: "View Certificate",
  },
  {
    title: "Best Graduate, Mechatronics Engineering",
    description:
      "Recognized as the top graduate of the Mechatronics Engineering program for outstanding academic performance and project excellence.",
    link: "https://drive.google.com/file/d/1e3byD04eSo2OzKgJsXillLh0GN1xpA76/view?usp=sharing",
    linkLabel: "View Certificate",
  },
]



export default function App() {
  return (
    <div className="container">
      <div className="nav">
        <div className="brandRow">
          <img className="avatar" src={profileImg} alt="Harold Lucero" />
          <div className="brand">
            <strong>Harold Lucero</strong>
            <span>Machine Learning Engineer, Computer Vision, NLP</span>
          </div>
        </div>
        <div className="actions">
          <a className="btn" href="https://github.com/Esteebaan23" target="_blank" rel="noreferrer">
            GitHub
          </a>
          <a className="btn" href="https://www.linkedin.com/in/harold-lucero-nieto-a70275259/" target="_blank" rel="noreferrer">
            LinkedIn
          </a>
          <a className="btn" href="mailto:esteebaan.lucero.23@gmail.com">
            Email
          </a>
        </div>
      </div>

      <div className="hero">
        <h1 className="h1">Applied ML, NLP and Computer Vision projects.</h1>
        <p className="sub">
          I build end-to-end machine learning systems, from data preparation and model training to rigorous evaluation, 
          spanning NLP, computer vision, biomedical imaging, and applied ML tooling.
        </p>

        <div className="pills">
          <span className="pill">PyTorch</span>
          <span className="pill">TensorFlow</span>
          <span className="pill">Transformers</span>
          <span className="pill">Computer Vision</span>
          <span className="pill">Object Detection</span>
          <span className="pill">Semantic Segmentation</span>
          <span className="pill">NLP</span>
          <span className="pill">RAG</span>
          <span className="pill">Reproducibility</span>
          <span className="pill">Evaluation</span>
          <span className="pill">Deployment</span>
        </div>
      </div>

      <div className="section">
        <h2>Featured Projects</h2>

        <div className="featureList">
          {FEATURED.map((p) => (
            <article className="featureCard" key={p.title}>
              <div className="featureText">
                <h3>{p.title}</h3>
                <p className="meta">{p.meta}</p>
                <ul>
                  {p.bullets.map((b, i) => (
                    <li key={i}>{b}</li>
                  ))}
                </ul>

                <div className="links">
                  {p.links.map((l) => (
                    <a key={l.label} className="taglink" href={l.href} target="_blank" rel="noreferrer">
                      {l.label}
                    </a>
                  ))}
                </div>
              </div>

              <div className="featureMedia">
                <img src={p.image} alt={p.alt} />
              </div>
            </article>
          ))}
        </div>
      </div>
      <div className="section">
  <h2>Awards and Recognition</h2>

  <div className="grid">
    {AWARDS.map((a) => (
      <div className="card" key={a.title}>
        <h3>{a.title}</h3>
        <p className="meta">{a.description}</p>

        {a.link && (
          <div className="links">
            <a
              className="taglink"
              href={a.link}
              target="_blank"
              rel="noreferrer"
            >
              {a.linkLabel || "View Certificate"}
            </a>
          </div>
        )}
      </div>
    ))}
  </div>
</div>


      <div className="section">
        <h2>Machine Learning Projects (University)</h2>

        <div className="carouselWrap">
  <button
    className="carouselBtn"
    onClick={() => document.getElementById("ml-carousel")?.scrollBy({ left: -420, behavior: "smooth" })}
    aria-label="Scroll left"
  >
    ◀
  </button>

  <div id="ml-carousel" className="carousel" role="region" aria-label="Machine learning projects carousel">
    {ML_PROJECTS.map((p) => (
      <div className="carouselItem" key={p.title}>
        <div className="card" style={{ height: "100%" }}>
          <h3>{p.title}</h3>
          <p className="meta">{p.meta}</p>
          <ul>
            {p.bullets.map((b, i) => (
              <li key={i}>{b}</li>
            ))}
          </ul>
          <div className="links">
            {p.links.map((l) => (
              <a key={l.label} className="taglink" href={l.href} target="_blank" rel="noreferrer">
                {l.label}
              </a>
            ))}
          </div>
        </div>
      </div>
    ))}
  </div>

  <button
    className="carouselBtn"
    onClick={() => document.getElementById("ml-carousel")?.scrollBy({ left: 420, behavior: "smooth" })}
    aria-label="Scroll right"
  >
    ▶
  </button>
</div>

      </div>

      <div className="footer">
        © {new Date().getFullYear()} Harold Lucero. Built with React + Vite, deployed on GitHub Pages.
      </div>
    </div>
  )
}
