import lpr from "../assets/lpr.png";
import xray from "../assets/xray.png";
import organoidsImg from "../assets/organoids.png";
import smallObjectsImg from "../assets/small-objects.png";
import mask from "../assets/mask.jpg";
import skin from "../assets/skin.jpg";
import lane from "../assets/lane_seg.png";
import plate from "../assets/plates.jpg";
const CV_PROJECTS = [
  {
    title: "Vehicle & License Plate Detection with YOLO26 + OCR",
    meta: "YOLO26, Computer Vision, OCR, ROI Filtering, FiftyOne",
    image: lpr,
    alt: "Vehicle and license plate detection system preview",
    bullets: [
      "Developed a vehicle and license plate detection pipeline using YOLO26 with confidence-based validation.",
      "Implemented ROI filtering and optional plate-to-vehicle association constraints to reduce false positives.",
      "Integrated OCR triggered only on high-confidence detections to optimize efficiency.",
      "Designed evaluation and error analysis workflows using FiftyOne."
    ],
    links: [{ label: "GitHub", href: "https://github.com/Esteebaan23/Vehicle-License-Plate-Detection-with-YOLO26" }],
  },
  {
    title: "Chest X-Ray Classification with YOLO26",
    meta: "YOLO26, Medical Imaging AI, FiftyOne, GCP Deployment",
    image: xray,
    alt: "Chest X-ray classification system preview",
    bullets: [
      "Built a YOLO-based chest X-ray classification pipeline (Normal vs Anomaly).",
      "Designed preprocessing/augmentation workflows for robustness.",
      "Used FiftyOne for failure analysis and dataset inspection.",
      "Deployed on GCP with auto-suspend for cost-efficient hosting."
    ],
    links: [
      { label: "Demo GCP", href: "https://xray-service-5372311531.us-central1.run.app/" },
      { label: "GitHub", href: "https://github.com/Esteebaan23/XRay_YOLO26" },
    ],
  },
  {
    title: "Organoids: Chamber Forming vs Non-Forming and Fluorescence Prediction",
    meta: "Biomedical CV, hybrid classifiers, image-to-image generation",
    image: organoidsImg,
    alt: "Organoids project preview",
    bullets: [
      "Built hybrid classifiers (ResNet50 + ViT) for chamber-forming vs non-forming organoids.",
      "Developed conditional image-to-image models (U-Net / ResUNet) for fluorescence prediction.",
      "Integrated FiftyOne to inspect misclassifications and compare model variants."
    ],
    links: [
      { label: "GitHub", href: "https://github.com/Esteebaan23/Organoid-Brightfield-to-Fluorescence-Image-Translation" },
      { label: "UNT Research Day Poster", href: "https://drive.google.com/file/d/1uLRmndkkjBWE9pTHy7koUVp7rzAS30i1/view" },
    ],
  },
  {
    title: "D2SO: Detecting Distant and Small Objects for Vision-Based Autonomous Systems",
    meta: "Small object detection, segmentation, autonomous driving vision",
    image: smallObjectsImg,
    alt: "Small object detection preview",
    bullets: [
      "Built detection/segmentation pipelines using YOLOv11, U-Net, and SAM.",
      "Evaluated performance on multi-class perception tasks (humans, vehicles, structures).",
      "Improved robustness for vision-based autonomy under real-world conditions."
    ],
    links: [
      { label: "GitHub", href: "https://github.com/Esteebaan23/Vehicle_Human_Object_Segmentation" },
      { label: "IEEE Publication", href: "https://ieeexplore.ieee.org/document/11071446" },
    ],
  },
  {
    title: "Car Driving Lane Segmentation",
    meta: "Semantic segmentation, computer vision",
    image: lane,
    bullets: [
      "Built a lane segmentation pipeline to extract drivable lane markings from road scenes using SAM2.",
      "Implemented training and inference workflows that support rapid iteration and visual inspection of segmentation results.",
      "Generated qualitative visualizations to analyze model behavior and highlight common failure modes."
    ],
    links: [{ label: "GitHub", href: "https://github.com/Esteebaan23/Car_Driving_Lane_Segmentation_SAM2" }],
  },
  {
    title: "Classification of U.S Licence Plates",
    meta: "Computer vision classification, model benchmarking",
    image: plate,
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
    image: skin,
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
    image: mask,
    bullets: [
      "Built a real-time face mask classification pipeline using OpenCV, MediaPipe for face detection, and MobileNetV2 for mask recognition.",
      "Integrated face localization with a MobileNetV2 classifier, achieving F1-scores of 96% for the “With Mask” class and 94% for the “Without Mask” class.",
      "Implemented a live visualization loop to inspect predictions and system behavior during real-time inference."
    ],
    links: [{ label: "GitHub", href: "https://github.com/Esteebaan23/Face-Mask-Detector" }],
  },
];

export default CV_PROJECTS;