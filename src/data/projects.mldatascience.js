
import tabular1 from "../assets/house.jpg"; 
import tabular2 from "../assets/student.jpg"; 
import tabular3 from "../assets/store.jpg";

const MLDS_PROJECTS = [
  {
    title: "House Prices Prediction",
    meta: "Regression, feature engineering, model selection, evaluation",
    image: tabular1,
    alt: "House prices regression preview",
    bullets: [
     "Built an end-to-end regression pipeline for housing price prediction, including data cleaning, encoding, and feature preparation.",
      "Applied feature engineering and multicollinearity analysis to improve XGBoost and Random Forest regression performance.",
      "Reduced RMSE by over $9,000 and improved R² by more than 4% through iterative model refinement and evaluation."
    ],
    links: [{ label: "GitHub", href: "https://github.com/Esteebaan23/House_Price_Prediction" }],
  },
  {
    title: "Student Performance Prediction",
    meta: "Data analysis and supervised learning, interpretable models",
    image: tabular2,
    alt: "Student performance prediction preview",
    bullets: [
      "Built a predictive workflow to analyze student academic performance using supervised learning and clustering techniques.",
      "Performed exploratory data analysis and preprocessing, training Linear Regression and Random Forest models with accuracies of 88% and 87%.",
      "Trained and tuned a neural network model, achieving 98.5% accuracy and identifying key factors influencing academic outcomes."
    ],
    links: [{ label: "GitHub", href: "https://github.com/Esteebaan23/Student-Performance-Factors-Analysis" }],
  },
  {
    title: "Time Series / Forecasting ",
    meta: "Time Series Forecasting, Regression, XGBoost",
    image: tabular3,
    alt: "Forecasting project preview",
    bullets: [
      "Designed a time series forecasting workflow combining SARIMA baselines with gradient-boosted regression models.",
      "Trained and optimized an XGBoost regressor, reaching an RMSE of 206.54 on held-out data.",
      "Created a lightweight local visualization tool to inspect forecast trends and model behavior."
    ],
    links: [{ label: "GitHub", href: "https://github.com/Esteebaan23/Store_Sales_Time_Series_XGboost" }],
  },
];

export default MLDS_PROJECTS;